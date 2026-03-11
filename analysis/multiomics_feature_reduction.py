import pandas as pd
import numpy as np
import warnings
import wandb
import os
import torch
from analysis.baselines import XGBoostClassifierWrapper, RealMLPClassifierWrapper
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.utils import PredictionResults
from analysis.data import get_wide_validation_datasets
from tabpfnwide.classifier import TabPFNWideClassifier

from tabicl import TabICLClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import warnings


warnings.filterwarnings("ignore")
import argparse


def main(
    dataset_name,
    checkpoint_paths,
    output_file,
    device="cuda:0",
    omics_list=None,
):
    """
    Runs feature reduction experiments on multi-omics datasets using a specified model and checkpoints.
    Parameters:
        dataset_name (str): Name of the dataset to evaluate.
        checkpoint_paths (list of str): List of paths to model checkpoints. Use "default" for the base model.
        output_file (str): Path to the CSV file where results will be saved.
        device (str, optional): Device to run the model on (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        omics_list (list of str, optional): List of omics types to include. If None, defaults to "mRNA".

    For each checkpoint and a range of feature counts, the function:
    - Loads the model and checkpoint.
    - Iterates over validation splits of the dataset with reduced features.
    - Evaluates the model, collects predictions, and computes accuracy and weighted F1 score.
    - Appends results to a CSV file, skipping experiments that have already been run.
    """

    api = wandb.Api()
    results = pd.DataFrame(
        columns=[
            "dataset_name",
            "omic",
            "checkpoint",
            "n_features",
            "fold",
            "accuracy",
            "roc_auc",
            "prediction_probas",
            "ground_truth",
        ]
    )
    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    for checkpoint_path in checkpoint_paths:
        print(f"Initializing model from {checkpoint_path}")
        if checkpoint_path == "v2" or checkpoint_path.startswith("wide-v2"):
            clf = TabPFNWideClassifier(
                model_name=checkpoint_path,
                device=device,
                ignore_pretraining_limits=True,
                save_attention_maps=False,
            )
            name = checkpoint_path
        elif checkpoint_path == "tabicl":
            clf = TabICLClassifier(device=device, n_estimators=1)
            name = "tabicl"
        elif checkpoint_path == "random_forest":
            clf = RandomForestClassifier(n_jobs=4)
            name = "random_forest"
        elif checkpoint_path == "xgboost":
            clf = XGBoostClassifierWrapper(device=device)
            name = "xgboost"
        elif checkpoint_path == "realmlp":
            clf = RealMLPClassifierWrapper(device=device)
            name = "realmlp"
        else:
            raise ValueError(f"Unknown checkpoint: {checkpoint_path}")

        for n_features in [
            200,
            500,
            2000,
            5000,
            7500,
            10000,
            15000,
            20000,
            25000,
            30000,
            0,
        ]:
            # Get already completed folds for this dataset+checkpoint+n_features combo
            completed_folds = set(
                results[
                    (results["n_features"] == n_features)
                    & (results["checkpoint"] == name)
                    & (results["dataset_name"] == dataset_name)
                ]["fold"].values
            )
            expected_folds = 5  # n_splits=5, n_repeats=1
            if len(completed_folds) == expected_folds:
                print(
                    f"Skipping {dataset_name} with {n_features} features, all {expected_folds} folds complete"
                )
                continue
            elif len(completed_folds) > 0:
                print(
                    f"Resuming {dataset_name} with {n_features} features: {len(completed_folds)}/{expected_folds} folds complete"
                )
            print(f"Validating {dataset_name} with {n_features} features")
            if hasattr(clf, "model") and clf.model is not None and hasattr(clf.model, "to"):
                clf.model.to(device)
                clf.model.eval()
            for i, dataset in enumerate(
                get_wide_validation_datasets(
                    device,
                    dataset_name=dataset_name,
                    n_splits=5,
                    n_repeats=1,
                    reduced_features=n_features,
                    omics=omics_list,
                )
            ):
                # Skip already completed folds
                if i in completed_folds:
                    continue

                X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = dataset

                # Convert tensors to numpy for TabPFNClassifier
                X_train = X_train_tensor.cpu().numpy()
                y_train = y_train_tensor.cpu().numpy().flatten()
                X_test = X_test_tensor.cpu().numpy()
                y_test = y_test_tensor.cpu().numpy().flatten()

                if checkpoint_path in ["tabicl", "random_forest", "xgboost", "realmlp"]:
                    if np.isnan(X_train).any():
                        imp = SimpleImputer(strategy="most_frequent")
                        X_train = imp.fit_transform(X_train)
                        X_test = imp.transform(X_test)

                clf.fit(X_train, y_train)
                pred_probs = clf.predict_proba(X_test)

                pred_res = PredictionResults(y_test, pred_probs)
                accuracy = pred_res.get_classification_report(print_report=False)["accuracy"]
                try:
                    if pred_probs.shape[-1] == 2:
                        roc_auc = roc_auc_score(y_test, pred_probs[:, 1])
                    else:
                        roc_auc = roc_auc_score(
                            y_test,
                            pred_probs,
                            multi_class="ovr",
                            average="macro",
                            labels=np.arange(pred_probs.shape[-1]),
                        )
                except Exception as e:
                    print(f"Error calculating ROC AUC: {e}")
                    roc_auc = np.nan

                results = pd.concat(
                    [
                        results,
                        pd.DataFrame(
                            {
                                "dataset_name": dataset_name,
                                "omic": "+".join(omics_list) if omics_list else "mRNA",
                                "checkpoint": name,
                                "n_features": X_train_tensor.shape[-1],
                                "fold": i,
                                "accuracy": accuracy,
                                "roc_auc": roc_auc,
                                "prediction_probas": [
                                    " ".join(map(str, pred_res.prediction_probas))
                                ],
                                "ground_truth": [" ".join(map(str, pred_res.ground_truth))],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            if hasattr(clf, "model") and hasattr(clf.model, "to"):
                clf.model.to("cpu")
            results.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="Path to data directory (unused but kept for compatibility)",
    )
    parser.add_argument("output_file", type=str)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--checkpoint_path", type=str)

    parser.add_argument("--omics", dest="omics_list", type=str, nargs="+", default=["mRNA"])
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    dataset_name = args.dataset
    output_file = args.output_file

    checkpoint_paths = []
    if args.checkpoint_path:
        checkpoint_paths.append(args.checkpoint_path)

    if not checkpoint_paths:
        checkpoint_paths = ["v2"]

    omics_list = args.omics_list
    device = args.device

    main(
        dataset_name,
        checkpoint_paths,
        output_file,
        device,
        omics_list,
    )
