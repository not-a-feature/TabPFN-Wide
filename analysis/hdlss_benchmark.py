import pandas as pd
import numpy as np
import os
import glob
from scipy.io import loadmat
from sklearn.utils import shuffle
import torch
import warnings

warnings.filterwarnings("ignore")
import sys

# Ensure the repository root is on sys.path so top-level packages (e.g. `tabpfnwide`)
# can be imported when this script is executed directly (python analysis/hdlss_benchmark.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from tabpfnwide.utils import PredictionResults
from tabpfnwide.patches import fit
from tabpfn.model_loading import load_model_criterion_config
from tabpfn import TabPFNClassifier

setattr(TabPFNClassifier, "fit", fit)
import argparse


def load_mat_file(mat_path):
    """
    Loads a MATLAB .mat file and extracts X and Y.

    Parameters:
        mat_path (str): Path to the .mat file.

    Returns:
        tuple: (X, y) numpy arrays, or (None, None) if loading fails.
    """
    try:
        data = loadmat(mat_path)

        # Extract X and Y from the loaded data
        if "X" in data and "Y" in data:
            X = data["X"]
            y = data["Y"].ravel()  # Flatten Y to 1D array
            return X, y
        else:
            print(f"Warning: 'X' or 'Y' not found in {mat_path}")
            return None, None
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None, None


def main(
    hdlss_data_dir,
    output_file,
    max_features=500,
    min_features=0,
    max_instances=10000,
    checkpoints=[],
    device="cuda:0",
):
    """
    Runs classification experiments on HDLSS datasets, evaluating different model checkpoints and the default model (TabPFNv2).

    Parameters:
        hdlss_data_dir (str): Path to the directory containing .mat files.
        output_file (str): Path to the CSV file where results will be saved or appended.
        max_features (int, optional): Maximum number of features allowed in a dataset. Defaults to 500.
        min_features (int, optional): Minimum number of features required in a dataset. Defaults to 0.
        max_instances (int, optional): Maximum number of instances allowed in a dataset. Defaults to 10,000.
        checkpoints (list, optional): List of checkpoint file paths to evaluate. Use "default" for the base model. Defaults to [].
        device (str, optional): Device identifier for model computation (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".

    Description:
        - Iterates over all .mat files in the specified directory.
        - Filters datasets based on feature/instance count and number of classes.
        - Evaluates TabPFNClassifier on each dataset using repeated stratified k-fold cross-validation.
        - Results are saved to the specified output CSV file.
    """

    # Get all .mat files from the directory
    mat_files = sorted(glob.glob(os.path.join(hdlss_data_dir, "*.mat")))

    if not mat_files:
        print(f"No .mat files found in {hdlss_data_dir}")
        return

    print(f"Found {len(mat_files)} datasets to process")

    for checkpoint_path in checkpoints:
        model = None

        if checkpoint_path != "default":
            print(f"Loading checkpoint: {checkpoint_path}")
            models, _, _, _ = load_model_criterion_config(
                model_path=None,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which="classifier",
                version="v2.5",
                download_if_not_exists=True,
            )
            model = models[0]
            model.features_per_group = 1
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Handle DDP-wrapped checkpoints
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Unwrap DDP prefix if present
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
        else:
            print("Using default TabPFNv2.5 model")

        res_df = pd.DataFrame(
            columns=[
                "dataset_name",
                "fold",
                "checkpoint",
                "accuracy",
                "f1_weighted",
                "roc_auc_score",
                "prediction_probas",
                "ground_truth",
            ]
        )

        if os.path.exists(output_file):
            res_df = pd.read_csv(output_file)

        for mat_file in mat_files:
            dataset_name = os.path.basename(mat_file).replace(".mat", "")

            # Check if this dataset has already been processed with this checkpoint
            if (
                (res_df["dataset_name"] == dataset_name)
                & (res_df["checkpoint"] == checkpoint_path.split("/")[-1])
            ).any():
                print(f"Skipping dataset {dataset_name}, already processed")
                continue

            try:
                X, y = load_mat_file(mat_file)

                if X is None or y is None:
                    print(f"Skipping {dataset_name}: Failed to load data")
                    continue

                # Filter by feature and instance count
                n_features = X.shape[1]
                n_instances = X.shape[0]
                n_classes = len(np.unique(y))

                if n_features < min_features or n_features > max_features:
                    print(
                        f"Skipping {dataset_name}: {n_features} features outside range [{min_features}, {max_features}]"
                    )
                    continue

                if n_instances > max_instances:
                    print(
                        f"Skipping {dataset_name}: {n_instances} instances exceeds max {max_instances}"
                    )
                    continue

                # TabPFN is not suitable for datasets with more than 10 classes
                if n_classes >= 10 or n_classes <= 1:
                    print(
                        f"Skipping {dataset_name}: {n_classes} classes outside suitable range (2-9)"
                    )
                    continue

                # Shuffle and encode labels
                X, y = shuffle(X, y, random_state=42)
                X = X.astype(np.float32)

                le = LabelEncoder()
                y = le.fit_transform(y)

                print(
                    f"Processing {dataset_name} (features: {n_features}, instances: {n_instances}, classes: {n_classes})"
                )

                clf = TabPFNClassifier(
                    device=device, n_estimators=1, ignore_pretraining_limits=True
                )
                skf = RepeatedStratifiedKFold(
                    n_splits=3, n_repeats=10 if X.shape[0] < 2500 else 3, random_state=42
                )

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    clf.fit(X_train, y_train, model=model if checkpoint_path != "default" else None)
                    pred_probs = clf.predict_proba(X_test)

                    pred_res = PredictionResults(y_test, pred_probs)
                    accuracy = pred_res.get_classification_report(print_report=False)["accuracy"]
                    f1_weighted = pred_res.get_f1_score(average="weighted")

                    if pred_probs.shape[-1] == 2:
                        roc_auc = roc_auc_score(
                            pred_res.ground_truth, pred_res.prediction_probas[:, 1]
                        )
                    else:
                        roc_auc = roc_auc_score(
                            pred_res.ground_truth,
                            pred_res.prediction_probas,
                            multi_class="ovr",
                            average="macro",
                            labels=np.arange(pred_probs.shape[-1]),
                        )

                    res_df = pd.concat(
                        [
                            res_df,
                            pd.DataFrame(
                                {
                                    "dataset_name": dataset_name,
                                    "fold": fold,
                                    "checkpoint": checkpoint_path.split("/")[-1],
                                    "accuracy": accuracy,
                                    "f1_weighted": f1_weighted,
                                    "roc_auc_score": roc_auc,
                                    "prediction_probas": [
                                        " ".join(map(str, pred_res.prediction_probas))
                                    ],
                                    "ground_truth": [" ".join(map(str, pred_res.ground_truth))],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

                print(f"Dataset {dataset_name} processed successfully.")
                res_df.to_csv(output_file, index=False)

            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TabPFN on HDLSS datasets")
    parser.add_argument("hdlss_data_dir", type=str, help="Path to directory containing .mat files")
    parser.add_argument("output_file", type=str, help="Path to output CSV file for results")
    parser.add_argument(
        "--max_features", type=int, default=50000, help="Maximum number of features to consider"
    )
    parser.add_argument(
        "--min_features", type=int, default=0, help="Minimum number of features to consider"
    )
    parser.add_argument(
        "--max_instances", type=int, default=10000, help="Maximum number of instances to consider"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to directory containing checkpoint files",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for computation (cuda:0 or cpu)"
    )

    args = parser.parse_args()

    # Load all checkpoints from directory
    checkpoints = [
        os.path.join(args.checkpoint_dir, f)
        for f in os.listdir(args.checkpoint_dir)
        if f.endswith(".pt")
    ]
    checkpoints += ["default"]

    main(
        hdlss_data_dir=args.hdlss_data_dir,
        output_file=args.output_file,
        max_features=args.max_features,
        min_features=args.min_features,
        max_instances=args.max_instances,
        checkpoints=checkpoints,
        device=args.device,
    )
