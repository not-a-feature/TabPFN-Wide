import pandas as pd
import numpy as np
import os
import glob
import torch
import warnings

import sys
import argparse
from analysis.baselines import XGBoostClassifierWrapper, RealMLPClassifierWrapper
import gc
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from pandas_plink import read_plink
from tabicl import TabICLClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Ensure the repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tabpfnwide.classifier import TabPFNWideClassifier


warnings.filterwarnings("ignore")


def select_snps_with_causal_in_haystack(genotypes, causal_snps, n_features=10000, random_state=42):
    """
    Select SNPs in a 'needle in a haystack' manner:
    - Always include all causal SNPs present in the genotype data
    - Randomly fill the remaining slots with non-causal (noise) SNPs up to n_features total
    """
    rng = np.random.default_rng(random_state)

    all_snps = np.array(genotypes.columns)
    causal_set = set(causal_snps.iloc[:, 0])

    # identify which causal SNPs actually appear in the genotype data
    causal_snps_in_data = np.array([s for s in all_snps if s in causal_set])
    non_causal_snps = np.array([s for s in all_snps if s not in causal_set])

    n_causal = len(causal_snps_in_data)

    # determine how many non-causal SNPs to add
    n_noncausal_target = max(0, n_features - n_causal)

    # sample noise SNPs
    selected_noncausal = rng.choice(
        non_causal_snps, size=min(n_noncausal_target, len(non_causal_snps)), replace=False
    )

    # combine causal + random noise SNPs
    selected_snps = np.concatenate([causal_snps_in_data, selected_noncausal])
    rng.shuffle(selected_snps)

    # also return the indices of the causal SNPs (within selected_snps)
    causal_indices = np.flatnonzero(np.isin(selected_snps, causal_snps_in_data))

    return genotypes[selected_snps], causal_snps_in_data, causal_indices


def main(
    data_dir,
    output_file,
    checkpoints=[],
    device="cuda:0",
    runs=[1, 2, 3],  # Default runs from original script
):

    plink_path = os.path.join(data_dir, "test_chr-1")
    if not os.path.exists(plink_path + ".bed"):
        print(f"Error: PLINK file not found at {plink_path}")
        return

    print(f"Loading Genotypes from {plink_path}...")
    (bim, fam, bed) = read_plink(plink_path)
    genotypes_full = bed.compute().T  # shape: (n_individuals, n_variants)
    genotypes_full = pd.DataFrame(genotypes_full, columns=bim.snp, index=fam.iid)
    print("Genotypes loaded.")

    res_df = pd.DataFrame(
        columns=[
            "dataset",
            "model",
            "run",
            "fold",
            "accuracy",
            "roc_auc",
            "n_features",
            "n_causal_snps",
            "n_samples",
            "Polygenicity",
        ]
    )

    if os.path.exists(output_file):
        try:
            res_df = pd.read_csv(output_file)
        except:
            pass
    # todo re increase
    feature_counts = [500, 1000, 5000, 10000, 15000, 20000, 30000, 50000, 60000, 70000]

    for model_name in checkpoints:
        print(f"Processing model: {model_name}")

        # Initialize model
        if model_name == "v2" or model_name.startswith("wide-v2"):
            clf = TabPFNWideClassifier(
                model_name=model_name,
                device=device,
                ignore_pretraining_limits=True,
                save_attention_maps=False,
            )
        elif model_name == "tabicl":
            clf = TabICLClassifier(device=device, n_estimators=1)
        elif model_name == "random_forest":
            clf = RandomForestClassifier(n_jobs=4)
        elif model_name == "xgboost":
            clf = XGBoostClassifierWrapper(device=device)
        elif model_name == "realmlp":
            clf = RealMLPClassifierWrapper(device=device)
        else:
            raise ValueError(f"Unknown checkpoint: {model_name}")

        # Ensure model is on device
        # clf.device = device # handled in init

        for run in runs:
            dataset_name = f"chr1_hapnest_run{run}"
            polygenicity = [0.001, 0.05, 0.01][run - 1]

            causal_file = os.path.join(data_dir, f"test_chr_run{run}.causal1")
            pheno_file = os.path.join(data_dir, f"test_chr_run{run}.pheno1")

            if not os.path.exists(causal_file) or not os.path.exists(pheno_file):
                print(f"Missing data files for run {run}, skipping.")
                continue

            causal_snps = pd.read_csv(causal_file, header=None, sep=r"\s+")
            pheno = pd.read_csv(pheno_file, sep=r"\s+")
            pheno.set_index("Sample", inplace=True)

            for n_f in feature_counts:
                model_identifier = model_name.split("/")[-1]

                # Get already completed folds for this dataset+model+n_features combo
                completed_folds = set(
                    res_df[
                        (res_df["dataset"] == dataset_name)
                        & (res_df["model"] == model_identifier)
                        & (res_df["n_features"] == n_f)
                    ]["fold"].values
                )

                print(f"  Run {run}, Features {n_f}")

                subset_genotypes, causal_snps_in_data, causal_indices = (
                    select_snps_with_causal_in_haystack(
                        genotypes_full,
                        causal_snps,
                        n_features=n_f,
                    )
                )

                num_causal_snps = len(causal_indices)

                df = subset_genotypes.join(pheno, how="inner")
                data = np.asarray(df)

                X = df.iloc[:, 1:-5].values
                # y = df.iloc[:, -1].values # Phenotype(binary) is last
                y = df["Phenotype(binary)"].astype(int).values

                # Cross validation
                strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                fold_accs = []
                fold_aucs = []

                expected_folds = strat_kf.get_n_splits()
                if len(completed_folds) == expected_folds:
                    print(
                        f"Skipping {dataset_name} with {n_f} features for {model_identifier}, all {expected_folds} folds complete"
                    )
                    continue
                elif len(completed_folds) > 0:
                    print(
                        f"Resuming {dataset_name} with {n_f} features: {len(completed_folds)}/{expected_folds} folds complete"
                    )

                for fold, (train_idx, test_idx) in enumerate(strat_kf.split(X, y)):
                    if fold + 1 in completed_folds:  # fold column uses 1-indexed values
                        continue

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    if model_name in ["tabicl", "random_forest", "xgboost", "realmlp"]:
                        if np.isnan(X_train).any():
                            imp = SimpleImputer(strategy="most_frequent")
                            X_train = imp.fit_transform(X_train)
                            X_test = imp.transform(X_test)

                    try:
                        clf.fit(X_train, y_train)
                        pred_probs = clf.predict_proba(X_test)

                        # Handle binary classification
                        if pred_probs.shape[1] == 2:
                            prob_score = pred_probs[:, 1]
                            y_pred = (prob_score > 0.5).astype(int)
                        else:
                            # Should be binary for this task
                            prob_score = pred_probs[:, 1]
                            y_pred = pred_probs.argmax(axis=1)

                        acc = accuracy_score(y_test, y_pred)
                        auc = roc_auc_score(y_test, prob_score)

                        fold_accs.append(acc)
                        fold_aucs.append(auc)

                        res_df = pd.concat(
                            [
                                res_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "dataset": dataset_name,
                                            "model": model_identifier,
                                            "run": run,
                                            "fold": fold + 1,
                                            "accuracy": acc,
                                            "roc_auc": auc,
                                            "n_features": n_f,
                                            "n_causal_snps": num_causal_snps,
                                            "n_samples": X.shape[0],
                                            "Polygenicity": polygenicity,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )

                    except Exception as e:
                        print(f"Error on fold {fold}: {e}")
                        import traceback

                        traceback.print_exc()

                # Clean up to save memory
                gc.collect()
                torch.cuda.empty_cache()

                # Update CSV after each feature set
                res_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TabPFN on SNP datasets")
    parser.add_argument(
        "data_dir", type=str, help="Path to directory containing SNP data (plink files, etc)"
    )
    parser.add_argument("output_file", type=str, help="Path to output CSV file for results")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint file or model name (e.g. 'v2')",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for computation (cuda:0 or cpu)"
    )

    args = parser.parse_args()

    # Determine checkpoints
    checkpoints = []
    if args.checkpoint_path:
        checkpoints.append(args.checkpoint_path)

    if not checkpoints:
        checkpoints = ["v2"]  # Default

    main(
        data_dir=args.data_dir,
        output_file=args.output_file,
        checkpoints=checkpoints,
        device=args.device,
    )
