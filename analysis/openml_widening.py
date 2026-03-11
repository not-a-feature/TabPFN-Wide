from sklearn.impute import SimpleImputer
import torch
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
import openml
from tabicl import TabICLClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from sklearn.utils import shuffle
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.utils import PredictionResults, get_new_features
from analysis.baselines import XGBoostClassifierWrapper, RealMLPClassifierWrapper
from tabpfnwide.classifier import TabPFNWideClassifier


import argparse


def main(
    did,
    output_file,
    device,
    checkpoint_path,
    sparsity=0.01,
    feature_numbers=[0, 50, 500, 2000, 5000, 10000, 20000, 30000],
):
    """
    Runs feature widening experiments on an OpenML dataset using various classifiers and checkpoints.
    Parameters:
        did (int): OpenML dataset ID to load.
        output_file (str): Path to the CSV file where results will be saved.
        device (str or torch.device): Device to use for model inference (e.g., 'cpu' or 'cuda').
        checkpoint_path (str): Checkpoint path or classifier name to evaluate.

        sparsity (float, optional): Sparsity level for added features. Default is 0.01.
        feature_numbers (list of int, optional): List of target feature counts for widening. Default is [0, 50, 500, 2000, 5000, 10000, 20000, 30000].
    Description:
        - Loads the specified OpenML dataset and preprocesses it (encoding, imputation, etc.).
        - For the checkpoint/classifier and each specified feature count:
            - Widens the dataset by adding new features with given sparsity.
            - Trains and evaluates the classifier using 5-fold cross-validation.
            - Computes accuracy, weighted F1, and ROC AUC scores.
            - Saves results (including predictions and ground truth) to the output CSV file.
    """

    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
    X, y = shuffle(X, y, random_state=42)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if sparsity != 0:
        X = X[X.columns[np.array(categorical_indicator) == False]]
        if did == 46940:  # Date column causing issues
            X = X.drop(columns=["Dt_Customer"])
        if X.isnull().values.any():
            simple_imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(simple_imputer.fit_transform(X), columns=X.columns)

    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame(
            columns=[
                "dataset_id",
                "fold",
                "checkpoint",
                "sparsity",
                "features",
                "features_added",
                "accuracy",
                "roc_auc_score",
                "prediction_probas",
                "ground_truth",
            ]
        )

    other_classifiers = ["tabicl", "random_forest", "xgboost", "realmlp"]
    clf = None
    if checkpoint_path == "tabicl":
        pass  # Will be created in loop
    elif checkpoint_path == "random_forest":
        pass  # Will be created in loop
    elif checkpoint_path == "xgboost":
        pass  # Will be created in loop
    elif checkpoint_path == "realmlp":
        pass  # Will be created in loop
    elif checkpoint_path == "v2" or checkpoint_path.startswith("wide-v2"):
        clf = TabPFNWideClassifier(
            model_name=checkpoint_path,
            device=device,
            ignore_pretraining_limits=True,
            save_attention_maps=False,
        )
        name = checkpoint_path
    else:
        raise ValueError(f"Unknown checkpoint: {checkpoint_path}")

    if checkpoint_path in ["tabicl", "random_forest", "xgboost", "realmlp"]:
        if X.isnull().values.any():
            simple_imputer = SimpleImputer(strategy="most_frequent")
            X = pd.DataFrame(simple_imputer.fit_transform(X), columns=X.columns)
            print(X.isnull().values.any())

        for col in X.columns:
            if X[col].dtype == object or X[col].dtype.name == "category":
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    for feature_number in feature_numbers:
        feature_number = int(feature_number)
        print(
            f"Validating {dataset.name} ({dataset.id}) with {feature_number} features and checkpoint {checkpoint_path}"
        )
        if checkpoint_path == "tabicl":
            clf = TabICLClassifier(device=device, n_estimators=1)
        elif checkpoint_path == "random_forest":
            clf = RandomForestClassifier(n_jobs=4)
        elif checkpoint_path == "xgboost":
            clf = XGBoostClassifierWrapper(device=device)
        elif checkpoint_path == "realmlp":
            clf = RealMLPClassifierWrapper(device=device)

        if feature_number != 0 and feature_number < X.shape[1]:
            print(
                f"Feature number {feature_number} is less than the original number of features {X.shape[1]}, skipping widening."
            )
            continue
        # Widen the dataset
        if sparsity == 0 and feature_number != 0:
            features_added = feature_number - X.shape[1]
        else:
            features_added = feature_number
        tensor = torch.Tensor(X.values) if sparsity != 0 else torch.zeros(X.shape)
        X_increase = get_new_features(tensor, features_added, sparsity, 1, False).numpy()
        if sparsity == 0 or feature_number == 0:
            X_new = np.concatenate((X.values, X_increase), axis=1)
        else:
            X_new = X_increase
        permuted_indices = np.random.permutation(X_new.shape[1])
        X_new = X_new[:, permuted_indices]

        assert X_new.shape[1] == (
            X.shape[1] if feature_number == 0 else feature_number
        ), f"Expected {feature_number} features, got {X_new.shape[1]}"
        check_feature_number = feature_number if feature_number != 0 else X.shape[1]
        exists = (
            (results["features"] == check_feature_number)
            & (
                results["checkpoint"].apply(
                    lambda x: x
                    in (
                        checkpoint_path.split("/")[-1]
                        if not (
                            checkpoint_path in ["default", "tabicl", "random_forest", "xgboost"]
                        )
                        else checkpoint_path
                    )
                )
            )
            & (results["dataset_id"] == did)
        ).any()
        if exists:
            print(f"Skipping dataset ID {did} with {X_new.shape[1]} features, already exists")
            continue

        for fold, (train_idx, test_idx) in enumerate(
            RepeatedStratifiedKFold(n_repeats=1, n_splits=5, random_state=42).split(X_new, y)
        ):
            X_train, X_test = X_new[train_idx], X_new[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train, y_train)
            pred_probs = clf.predict_proba(X_test)
            pred_res = PredictionResults(y_test, pred_probs)
            accuracy = pred_res.get_classification_report(print_report=False)["accuracy"]
            if pred_probs.shape[-1] == 2:
                roc_auc = roc_auc_score(pred_res.ground_truth, pred_res.prediction_probas[:, 1])
            else:
                roc_auc = roc_auc_score(
                    pred_res.ground_truth,
                    pred_res.prediction_probas,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(pred_probs.shape[-1]),
                )
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {
                            "dataset_id": did,
                            "fold": fold,
                            "checkpoint": (
                                checkpoint_path.split("/")[-1].replace(".pt", "")
                                if checkpoint_path not in other_classifiers
                                else checkpoint_path
                            ),
                            "sparsity": sparsity,
                            "features": X_new.shape[1],
                            "features_added": features_added,
                            "accuracy": accuracy,
                            "roc_auc_score": roc_auc,
                            "prediction_probas": [" ".join(map(str, pred_res.prediction_probas))],
                            "ground_truth": [" ".join(map(str, pred_res.ground_truth))],
                        }
                    ),
                ],
                ignore_index=True,
            )

        results.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", type=str)
    parser.add_argument("--dataset_ids", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument(
        "--feature_numbers",
        type=int,
        nargs="+",
        default=[0, 50, 500, 2000, 5000, 10000, 20000, 30000],
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    for dataset_id in args.dataset_ids:
        output_file = os.path.join(args.output_folder, f"{dataset_id}.csv")
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        try:
            main(
                dataset_id,
                output_file,
                args.device,
                args.checkpoint_path,
                sparsity=args.sparsity,
                feature_numbers=args.feature_numbers,
            )
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")
            continue
