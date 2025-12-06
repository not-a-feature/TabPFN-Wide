import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import os
import glob
from scipy.io import loadmat
from sklearn.utils import shuffle
import torch
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tabpfnwide.utils import PredictionResults
from tabpfnwide.patches import fit
from tabpfn.model_loading import load_model_criterion_config
from tabpfn import TabPFNClassifier
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

setattr(TabPFNClassifier, "fit", fit)


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


def plot_results(results_file, output_plot):
    """Generate comparison plot of performance across grouping settings."""
    df = pd.read_csv(results_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["accuracy", "f1_weighted", "roc_auc_score"]
    
    for idx, metric in enumerate(metrics):
        # Aggregate across folds for each task
        agg_df = df.groupby(["task_id", "features_per_group"])[metric].mean().reset_index()
        
        sns.boxplot(data=agg_df, x="features_per_group", y=metric, ax=axes[idx])
        axes[idx].set_title(f"{metric.replace('_', ' ').title()}")
        axes[idx].set_xlabel("Features Per Group")
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    summary = df.groupby("features_per_group")[metrics].agg(["mean", "std"])
    print(summary)


def main(
    hdlss_data_dir,
    output_file,
    max_features=100,
    min_features=0,
    max_instances=2000,
    grouping_values=[1, 2, 3],
    device="cuda:0",
    generate_plot=True,
):
    """
    Benchmark TabPFN base model with different features_per_group settings using HDLSS datasets.
    
    Parameters:
        hdlss_data_dir (str): Path to the directory containing .mat files.
        output_file (str): Path to the CSV file where results will be saved or appended.
        max_features (int, optional): Maximum number of features allowed in a dataset. Defaults to 100.
        min_features (int, optional): Minimum number of features required in a dataset. Defaults to 0.
        max_instances (int, optional): Maximum number of instances allowed in a dataset. Defaults to 2000.
        grouping_values (list, optional): List of features_per_group values to test. Defaults to [1, 2, 3].
        device (str, optional): Device identifier for model computation. Defaults to "cuda:0".
        generate_plot (bool, optional): Whether to generate comparison plot. Defaults to True.
    """
    # Get all .mat files from the directory
    if not os.path.exists(hdlss_data_dir):
        print(f"Error: Directory {hdlss_data_dir} does not exist")
        return

    mat_files = sorted(glob.glob(os.path.join(hdlss_data_dir, "*.mat")))
    
    if not mat_files:
        print(f"No .mat files found in {hdlss_data_dir}")
        return

    print(f"Found {len(mat_files)} datasets to process")
    print(f"Testing features_per_group values: {grouping_values}")

    # Initialize results DataFrame
    res_df = pd.DataFrame(
        columns=[
            "task_id", # Keeping column name for compatibility, will use dataset name
            "task_name",
            "num_features",
            "num_instances",
            "num_classes",
            "fold",
            "features_per_group",
            "accuracy",
            "f1_weighted",
            "roc_auc_score",
        ]
    )
    
    if os.path.exists(output_file):
        res_df = pd.read_csv(output_file)

    for grouping in grouping_values:
        print(f"\n{'='*60}")
        print(f"Testing features_per_group = {grouping}")
        print(f"{'='*60}")
        
        # Load base TabPFN v2.5 model
        models, _, _, _ = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which="classifier",
            version="v2.5",
            download_if_not_exists=True,
        )
        model = models[0]
        model.features_per_group = grouping

        for mat_file in mat_files:
            dataset_name = os.path.basename(mat_file).replace(".mat", "")
            
            # Check if already processed
            if (
                (res_df["task_name"] == dataset_name)
                & (res_df["features_per_group"] == grouping)
            ).any():
                print(f"Skipping dataset {dataset_name} (grouping={grouping}), already processed")
                continue

            try:
                X, y = load_mat_file(mat_file)
                
                if X is None or y is None:
                    continue

                # Filter by feature and instance count
                num_features = X.shape[1]
                num_instances = X.shape[0]
                num_classes = len(np.unique(y))

                if num_features < min_features or num_features > max_features:
                    # print(f"Skipping {dataset_name}: {num_features} features outside range")
                    continue
                
                if num_instances > max_instances:
                    # print(f"Skipping {dataset_name}: {num_instances} instances exceeds max")
                    continue

                if num_classes < 2 or num_classes > 10:
                    # print(f"Skipping {dataset_name}: {num_classes} classes not supported")
                    continue

                print(f"\nDataset: {dataset_name}")
                print(f"  Features: {num_features}, Instances: {num_instances}, Classes: {num_classes}")

                X, y = shuffle(X, y, random_state=42)
                X = X.astype(np.float32)
                le = LabelEncoder()
                y = le.fit_transform(y)

                clf = TabPFNClassifier(
                    device=device, n_estimators=1, ignore_pretraining_limits=True
                )
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    clf.fit(X_train, y_train, model=model)
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
                                    "task_id": [dataset_name], # Using dataset name as ID
                                    "task_name": [dataset_name],
                                    "num_features": [num_features],
                                    "num_instances": [num_instances],
                                    "num_classes": [num_classes],
                                    "fold": [fold],
                                    "features_per_group": [grouping],
                                    "accuracy": [accuracy],
                                    "f1_weighted": [f1_weighted],
                                    "roc_auc_score": [roc_auc],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                    print(f"  Fold {fold}: accuracy={accuracy:.3f}, f1={f1_weighted:.3f}, roc_auc={roc_auc:.3f}")

                # Save results after each task
                res_df.to_csv(output_file, index=False)
                print(f"Dataset {dataset_name} processed successfully (grouping={grouping})")

            except Exception as e:
                print(f"Error with dataset {dataset_name}: {e}")
                continue

    print(f"Results saved to {output_file}")

    if generate_plot and len(res_df) > 0:
        plot_output = output_file.replace(".csv", "_plot.png")
        plot_results(output_file, plot_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark TabPFN with different features_per_group settings on HDLSS data"
    )
    parser.add_argument("hdlss_data_dir", type=str, 
                       help="Path to directory containing .mat files")
    parser.add_argument(
        "--max_features", type=int, default=100000, 
        help="Maximum number of features to consider"
    )
    parser.add_argument(
        "--min_features", type=int, default=0, 
        help="Minimum number of features to consider"
    )
    parser.add_argument(
        "--max_instances", type=int, default=10000, 
        help="Maximum number of instances to consider"
    )
    parser.add_argument(
        "--output_file", type=str, default="grouping_benchmark_hdlss_results.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--grouping_values", type=int, nargs="+", default=[1, 2, 3],
        help="List of features_per_group values to test"
    )
    parser.add_argument("--device", type=str, default="cuda:0", 
                       help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--no_plot", action="store_true", 
                       help="Skip generating comparison plot")
    
    args = parser.parse_args()
    
    main(
        hdlss_data_dir=args.hdlss_data_dir,
        output_file=args.output_file,
        max_features=args.max_features,
        min_features=args.min_features,
        max_instances=args.max_instances,
        grouping_values=args.grouping_values,
        device=args.device,
        generate_plot=not args.no_plot,
    )
