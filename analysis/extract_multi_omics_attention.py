import os
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.load_mm_data import load_multiomics
from tabpfnwide.classifier import TabPFNWideClassifier
import warnings

warnings.filterwarnings("ignore")
import argparse


def main(
    dataset_name,
    output_file,
    checkpoint_path,
    device="cuda:0",
    omic="mrna",
    subsample_features=None,
):
    """
    Extracts and saves attention maps from a trained transformer-based model on multi-omics data.
    Parameters:
        dataset_name (str): Name of the multi-omics dataset to use.
        output_file (str): Path to save the extracted attention maps (as a torch file).
        checkpoint_path (str): Path to the model checkpoint to load weights from.

        device (str, optional): Device to run the model on (default: "cuda:0").
        omic (str, optional): Omics data type to use from the dataset (default: "mrna").
    Description:
        - Loads multi-omics data and encodes labels.
        - Loads a TabPFN-Wide model.
        - Configures the model to save attention maps during inference.
        - Runs inference to obtain predictions and attention maps.
        - Saves the extracted attention maps to the specified output file.
    """
    ds_dict, labels = load_multiomics(dataset_name)
    mrna = ds_dict[omic]
    X, y = mrna.values, labels
    y = LabelEncoder().fit_transform(y)
    print(X.shape)

    if checkpoint_path == "random_forest" or checkpoint_path == "tabicl":
        return

    clf = TabPFNWideClassifier(
        model_name=checkpoint_path,
        device=device,
        n_estimators=1,
        features_per_group=1,
        save_attention_maps=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    avg_maps = []

    if subsample_features is None:
        clf.fit(X_train, y_train)
        clf.predict_proba(X_test)
        avg_maps = clf.get_attention_maps().mean(axis=0)
    else:
        n_features = X.shape[1]
        for i in range(0, n_features, subsample_features):
            end = min(i + subsample_features, n_features)
            print(f"Processing features {i} to {end}...")

            X_train_sub = X_train[:, i:end]
            X_test_sub = X_test[:, i:end]

            clf.fit(X_train_sub, y_train)
            clf.predict_proba(X_test_sub)

            # Get attention maps and average over layers/heads for this chunk
            chunk_maps = clf.get_attention_maps().mean(axis=0)
            avg_maps.append(chunk_maps)

        avg_map = np.concatenate(avg_maps, axis=0)

    # Get feature names
    importance = avg_map.sum(axis=0)  # Importance per feature
    top_indices = np.argsort(importance)[::-1][:20]
    top_20_feature_names = mrna.columns[top_indices].tolist()

    # Save to file
    with open(f"{output_file}.txt", "w") as f:
        for feature_name, importance_score in zip(top_20_feature_names, importance[top_indices]):
            f.write(f"{feature_name}: {importance_score}\n")

    maps = torch.from_numpy(avg_map)
    torch.save(maps, f"{output_file}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="Path to data directory (unused but kept for compatibility)",
    )
    parser.add_argument("output_file", type=str, help="Path to save the attention maps")
    parser.add_argument(
        "--dataset", dest="dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the model checkpoint"
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--omic", type=str, default="mrna", help="Omic type to use (default: mRNA)")
    parser.add_argument(
        "--subsample_features",
        type=int,
        default=None,
        help="Number of features to process at a time (default: None = all features)",
    )

    args = parser.parse_args()
    main(
        args.dataset_name,
        args.output_file,
        args.checkpoint_path,
        args.device,
        args.omic,
        args.subsample_features,
    )
