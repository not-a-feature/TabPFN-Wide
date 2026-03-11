import argparse
import os
import pickle
import sys
import warnings

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from openml import tasks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier
from tabpfnwide.classifier import TabPFNWideClassifier
from analysis.utils import PredictionResults

# Filter warnings
warnings.filterwarnings("ignore")

# Column definitions
RESULT_COLUMNS = [
    "task_id",
    "task_name",
    "scenario_name",
    "num_features",
    "num_instances",
    "num_classes",
    "fold",
    "features_per_group",
    "duplicate_factor",
    "masks_injected",
    "n_estimators",
    "accuracy",
    "roc_auc_score",
    "analysis_type",
]

# Define scenarios configuration
SCENARIOS = [
    {"name": "A_n1", "grouping": 1, "dup": 1, "mask": 0, "type": "grouping", "n_estimators": 1},
    {"name": "A_n8", "grouping": 1, "dup": 1, "mask": 0, "type": "grouping", "n_estimators": 8},
    {"name": "B_n1", "grouping": 2, "dup": 1, "mask": 0, "type": "grouping", "n_estimators": 1},
    {"name": "B_n8", "grouping": 2, "dup": 1, "mask": 0, "type": "grouping", "n_estimators": 8},
    {"name": "D_n1", "grouping": 2, "dup": 2, "mask": 0, "type": "duplication", "n_estimators": 1},
    {"name": "D_n8", "grouping": 2, "dup": 2, "mask": 0, "type": "duplication", "n_estimators": 8},
    {"name": "F_n1", "grouping": 2, "dup": 1, "mask": 1, "type": "masking", "n_estimators": 1},
    {"name": "F_n8", "grouping": 2, "dup": 1, "mask": 1, "type": "masking", "n_estimators": 8},
]


def shuffle_arrays(*arrays, random_state=None):
    """Shuffle arrays in unison using a deterministic RNG."""
    if not arrays:
        return arrays

    length = arrays[0].shape[0]
    for arr in arrays:
        assert len(arr) == length, "All arrays must have the same length"

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(length)
    return tuple(arr[perm] for arr in arrays)


def print_all_results(df):
    """Print the consolidated benchmarks and a grouped summary.

    Args:
        df: Concatenated results from all enabled analyses.
    """

    if df.empty:
        print("No benchmark results to display.")
        return

    # Create a working copy to avoid modifying the original df in place if not desired
    df_work = df.copy()

    # Ensure scenario_name is populated for sorting
    if "scenario_name" not in df_work.columns:
        df_work["scenario_name"] = None

    def fill_scenario(row):
        if pd.notna(row["scenario_name"]):
            return row["scenario_name"]
        n_est = row["n_estimators"]
        for s in SCENARIOS:
            if (
                (s["grouping"] == row["features_per_group"])
                and (s["dup"] == row["duplicate_factor"])
                and (s["mask"] == row["masks_injected"])
                and (s["n_estimators"] == n_est)
            ):
                return s["name"]
        return "Z"  # End of list

    df_work["scenario_name"] = df_work.apply(fill_scenario, axis=1)

    sort_cols = ["scenario_name", "task_id"]
    df_sorted = df_work.sort_values(sort_cols).reset_index(drop=True)

    print("\n=== Complete Benchmark Results ===")
    print(df_sorted.to_string(index=False))

    metrics = ["accuracy", "f1_weighted", "roc_auc_score"]
    available_metrics = [m for m in metrics if m in df.columns]
    if available_metrics:
        summary = (
            df_sorted.groupby(
                [
                    "scenario_name",
                    "analysis_type",
                    "features_per_group",
                    "duplicate_factor",
                    "masks_injected",
                    "n_estimators",
                ]
            )[available_metrics]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        print("\n=== Aggregated Summary by Combination ===")
        print(summary.to_string(index=False))


def get_combo_label(row):
    s_name = f"Gr:{row['features_per_group']} Du:{row['duplicate_factor']} Ma:{row['masks_injected']} NEst:{row['n_estimators']}"
    return s_name


def plot_combined_results(df, output_plot):
    """Create a box plot that compares each combination across all metrics."""

    if df.empty:
        print("No results available for plotting.")
        return

    plot_df = df.copy()
    plot_df["duplicate_factor"] = plot_df["duplicate_factor"].fillna(1).astype(int)
    plot_df["masks_injected"] = plot_df["masks_injected"].fillna(0).astype(int)
    plot_df["features_per_group"] = plot_df["features_per_group"].fillna(1).astype(int)
    plot_df["n_estimators"] = plot_df["n_estimators"].fillna(8).astype(int)

    plot_df["combo_label"] = plot_df.apply(get_combo_label, axis=1)

    # Determine order
    unique_labels = (
        plot_df.groupby("combo_label")["accuracy"].median().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(max(10, len(unique_labels) * 0.45), 6))
    sns.boxplot(
        data=plot_df, x="combo_label", y="accuracy", ax=ax, palette="Set2", order=unique_labels
    )
    ax.set_title("Accuracy Distribution per Combination")
    ax.set_xlabel("Combination")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Calculate and annotate the mean on the box plot
    means = plot_df.groupby("combo_label")["accuracy"].mean()
    for i, label in enumerate(unique_labels):
        mean_value = means[label]
        ax.text(
            i,
            mean_value - 0.05,
            f"{mean_value:.4f}",
            horizontalalignment="center",
            color="black",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved to {output_plot}")


def plot_combined_auroc_results(df, output_plot):
    """Create a box plot that compares each combination across AUROC."""

    if df.empty:
        print("No results available for plotting.")
        return

    plot_df = df.copy()
    plot_df["duplicate_factor"] = plot_df["duplicate_factor"].fillna(1).astype(int)
    plot_df["masks_injected"] = plot_df["masks_injected"].fillna(0).astype(int)
    plot_df["features_per_group"] = plot_df["features_per_group"].fillna(1).astype(int)
    plot_df["n_estimators"] = plot_df["n_estimators"].fillna(8).astype(int)

    plot_df["combo_label"] = plot_df.apply(get_combo_label, axis=1)

    # Determine order
    unique_labels = (
        plot_df.groupby("combo_label")["roc_auc_score"].median().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(max(10, len(unique_labels) * 0.45), 6))
    sns.boxplot(
        data=plot_df, x="combo_label", y="roc_auc_score", ax=ax, palette="Set2", order=unique_labels
    )
    ax.set_title("AUROC Distribution per Combination")
    ax.set_xlabel("Combination")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Calculate and annotate the mean on the box plot
    means = plot_df.groupby("combo_label")["roc_auc_score"].mean()
    for i, label in enumerate(unique_labels):
        mean_value = means[label]
        ax.text(
            i,
            mean_value - 0.05,
            f"{mean_value:.4f}",
            horizontalalignment="center",
            color="black",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"AUROC plot saved to {output_plot}")


def register_embedding_hook(model, embeddings_list):
    """
    Register a forward hook on the input encoder to capture embeddings.

    Args:
        model: The TabPFN model
        embeddings_list: List to store captured embeddings

    Returns:
        Hook handle (for removal later)
    """

    def hook_fn(module, input, output):
        embeddings_list.append(output.detach().cpu().clone())

    # Hook into the encoder output
    # The encoder is model.encoder, which is a SequentialEncoder
    # We want to capture the output after the LinearInputEncoderStep
    assert hasattr(model, "encoder"), "Model must have encoder attribute"

    handle = model.encoder.register_forward_hook(hook_fn)
    return handle


def extract_embeddings_with_model(model, X_train, y_train, X_test, clf, device):
    """
    Extract embeddings from the model using a forward hook.

    Returns:
        embeddings: numpy array of embeddings from test set
        predictions: prediction probabilities
    """
    embeddings_list = []
    hook_handle = register_embedding_hook(model, embeddings_list)

    try:
        # Run forward pass
        clf.fit(X_train, y_train)
        pred_probs = clf.predict_proba(X_test)

        # Extract embeddings
        # The hook captures embeddings in the format from encoder output
        # We need the test portion
        if len(embeddings_list) > 0:
            # Get the last captured embeddings
            emb = embeddings_list[-1]
            # emb shape: (seq_len, batch*features, emsize)
            # We need to extract test portion and reshape appropriately
            embeddings = emb.numpy()
        else:
            embeddings = None

    finally:
        hook_handle.remove()

    return embeddings, pred_probs


def analyze_embeddings(embeddings_dict, output_dir, all_labels_order=None):
    """
    Analyze and visualize embeddings from different methods.

    Args:
        embeddings_dict: Dict mapping method name -> embeddings array
        output_dir: Directory to save visualizations
        all_labels_order: Optional list of all possible labels to ensure consistent coloring
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for UMAP
    all_embeddings = []
    all_labels = []

    for method_name, emb_array in embeddings_dict.items():
        if emb_array is None or len(emb_array) == 0:
            continue

        # Flatten to 2D if needed
        if len(emb_array.shape) > 2:
            # Take mean over sequence dimension
            emb_flat = emb_array.mean(axis=0)
        else:
            emb_flat = emb_array

        # Flatten further if needed
        if len(emb_flat.shape) > 2:
            emb_flat = emb_flat.reshape(emb_flat.shape[0], -1)

        all_embeddings.append(emb_flat)
        all_labels.extend([method_name] * len(emb_flat))

    if len(all_embeddings) == 0:
        print("No embeddings to analyze")
        return

    all_embeddings = np.vstack(all_embeddings)

    # Standardize
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings)

    # UMAP projection
    print("Computing UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(all_embeddings_scaled)

    # Plot UMAP with different markers for different feature types
    fig, ax = plt.subplots(figsize=(12, 9))

    unique_methods = sorted(list(embeddings_dict.keys()))

    if all_labels_order:
        # Use consistent palette based on provided order
        palette = sns.color_palette("Set2", len(all_labels_order))
        color_map = dict(zip(all_labels_order, palette))
    else:
        # Fallback to local palette
        palette = sns.color_palette("Set2", len(unique_methods))
        color_map = dict(zip(unique_methods, palette))

    # Define marker mapping based on feature type
    def get_marker_for_method(method_name):
        """Determine marker based on method characteristics."""
        # Parse method name to extract features
        if "mask" in method_name.lower() or any(s in method_name for s in ["F_", "G_"]):
            return "X"  # X for masking
        elif "dup" in method_name.lower() or any(s in method_name for s in ["D_", "E_"]):
            return "s"  # square for duplication
        elif "mixed" in method_name.lower() or "H_" in method_name:
            return "D"  # diamond for mixed
        else:
            return "o"  # circle for grouping only

    for i, method in enumerate(unique_methods):
        mask = np.array(all_labels) == method
        color = color_map[method]
        marker = get_marker_for_method(method)

        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[color],
            marker=marker,
            label=method,
            alpha=0.6,
            s=80,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "UMAP Projection of Embeddings by Method\n(○:grouping, □:duplication, ×:masking, ◇:mixed)"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    umap_path = os.path.join(output_dir, "umap_embeddings.png")
    plt.savefig(umap_path, dpi=300, bbox_inches="tight")
    print(f"UMAP plot saved to {umap_path}")
    plt.close()

    # Distribution plots
    fig, axes = plt.subplots(1, len(unique_methods), figsize=(5 * len(unique_methods), 4))
    if len(unique_methods) == 1:
        axes = [axes]

    for i, method in enumerate(unique_methods):
        emb = embeddings_dict[method]
        if emb is not None and len(emb) > 0:
            emb_flat = emb.flatten()
            axes[i].hist(emb_flat, bins=50, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"{method}\nMean: {emb_flat.mean():.3f}, Std: {emb_flat.std():.3f}")
            axes[i].set_xlabel("Embedding Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "embedding_distributions.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    print(f"Distribution plot saved to {dist_path}")
    plt.close()

    # Statistics table
    stats_data = []
    for method in unique_methods:
        emb = embeddings_dict[method]
        if emb is not None and len(emb) > 0:
            emb_flat = emb.flatten()
            stats_data.append(
                {
                    "method": method,
                    "mean": emb_flat.mean(),
                    "std": emb_flat.std(),
                    "min": emb_flat.min(),
                    "max": emb_flat.max(),
                    "shape": str(emb.shape),
                }
            )

    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(output_dir, "embedding_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistics saved to {stats_path}")
    print("\n=== Embedding Statistics ===")
    print(stats_df.to_string(index=False))


def evaluate_task(
    task_id,
    grouping,
    clf,
    device,
    openml_task,
    res_df,
    duplicate_factor=1,
    masks_injected=0,
    n_estimators=8,
    extract_embeddings=False,
    analysis_type="grouping",
    scenario_name=None,
):
    """
    Evaluates a single task with specific grouping and feature duplication settings.

    Args:
        duplicate_factor: Total number of copies of each feature (1 = no duplication).
        masks_injected: Number of mask columns to inject after the feature copies.
        n_estimators: Number of estimators for TabPFNClassifier.

    Returns:
        res_df: Updated results dataframe
        embeddings: Extracted embeddings (if extract_embeddings=True), else None
    """
    dataset = openml_task.get_dataset()
    X, y, _, _ = dataset.get_data(target=openml_task.target_name)

    # Ensure X is numpy array for consistent manipulation
    if hasattr(X, "values"):
        X = X.values
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X)

    # Transformation logic
    # We want for each feature f: [f, f... (duplicate_factor), mask... (masks_injected)]

    expansion_size = duplicate_factor + masks_injected

    if expansion_size > 1:
        n_samples, n_features = X.shape

        # Reshape X to (N, F, 1)
        X_expanded = np.expand_dims(X, axis=2)

        # Repeat for duplication: (N, F, duplicate_factor)
        X_duplicated = np.repeat(X_expanded, duplicate_factor, axis=2)

        # Create the full array with NaNs
        X_new_reshaped = np.full((n_samples, n_features, expansion_size), np.nan, dtype=X.dtype)

        # Fill the first duplicate_factor slots
        X_new_reshaped[:, :, :duplicate_factor] = X_duplicated

        # Flatten back to (N, F * expansion_size)
        X = X_new_reshaped.reshape(n_samples, n_features * expansion_size)

    df_sample = pd.DataFrame(X)
    print("Sample of processed X (first 3 rows, up to 10 cols):")
    print(df_sample.iloc[:3, :10].to_string(index=False))

    # Assertions for validity
    assert len(X) == len(y), "X and y must have same number of instances"
    assert duplicate_factor >= 1, "Duplicate factor must be >= 1"

    num_features = X.shape[1]
    num_instances = X.shape[0]
    num_classes = len(np.unique(y))
    dataset_name = dataset.name

    print(
        f"\nTask: {task_id} (Dataset: {dataset_name}) - Grouping: {grouping}, Dup: {duplicate_factor}, Masks: {masks_injected}, N_est: {n_estimators}"
    )
    print(f"  Features: {num_features}, Instances: {num_instances}, Classes: {num_classes}")

    # Shuffle
    X, y = shuffle_arrays(X, y, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    skf = RepeatedStratifiedKFold(
        n_splits=3,
        n_repeats=10 if X.shape[0] < 2500 else 3,
        random_state=42,
    )

    embeddings_collected = [] if extract_embeddings else None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            if extract_embeddings and fold == 0:  # Only extract for first fold
                embeddings, pred_probs = extract_embeddings_with_model(
                    clf.model, X_train, y_train, X_test, clf, device
                )
                if embeddings is not None:
                    embeddings_collected.append(embeddings)
            else:
                clf.fit(X_train, y_train)
                pred_probs = clf.predict_proba(X_test)

            # Verify predictions shape
            assert pred_probs.shape[0] == len(
                y_test
            ), f"Predictions shape mismatch: {pred_probs.shape} vs {len(y_test)}"

            pred_res = PredictionResults(y_test, pred_probs)
            report = pred_res.get_classification_report(print_report=False)
            accuracy = report["accuracy"]

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

            row_data = {
                "task_id": [task_id],
                "task_name": [dataset_name],
                "scenario_name": [scenario_name],
                "num_features": [num_features],
                "num_instances": [num_instances],
                "num_classes": [num_classes],
                "fold": [fold],
                "features_per_group": [grouping],
                "duplicate_factor": [duplicate_factor],
                "masks_injected": [masks_injected],
                "n_estimators": [n_estimators],
                "accuracy": [accuracy],
                "roc_auc_score": [roc_auc],
                "analysis_type": [analysis_type],
            }

            new_row = pd.DataFrame(row_data)
            res_df = pd.concat([res_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Skipping fold {fold} for task {task_id} due to error: {e}")
            continue

    final_embeddings = embeddings_collected[0] if embeddings_collected else None
    return res_df, final_embeddings


def main(
    suite_id,
    output_file,
    max_features=100,
    min_features=0,
    max_instances=2000,
    grouping_values=[1, 2, 3],
    device="cuda:0",
    generate_plot=True,
    duplication_output_file=None,
    masking_output_file=None,
    extract_embeddings=False,
    tasks_limit=None,
):
    """
    Benchmark TabPFN base model.
    """
    # Fetch OpenML suite
    suite = openml.study.get_suite(suite_id=suite_id)
    openml_df = tasks.list_tasks(output_format="dataframe", task_id=suite.tasks)
    print(f"Loaded tasks dataframe shape: {openml_df.shape}")

    assert not openml_df.empty, f"No tasks found for suite {suite_id}"

    if "task_type" in openml_df.columns:
        openml_df = openml_df[openml_df["task_type"] == "Supervised Classification"]

    openml_df = openml_df[
        (openml_df["NumberOfFeatures"] >= min_features)
        & (openml_df["NumberOfFeatures"] <= max_features)
        & (openml_df["NumberOfInstances"] <= max_instances)
        & (openml_df["NumberOfClasses"] < 10)
        & (openml_df["NumberOfClasses"] > 1)
    ]

    if tasks_limit is not None:
        print(f"Limiting to {tasks_limit} tasks")
        openml_df = openml_df.head(tasks_limit)

    print(f"Found {len(openml_df)} tasks to process in suite {suite_id}")

    if len(openml_df) == 0:
        print("No tasks meet the criteria. Exiting.")
        return

    # Model will be loaded via TabPFNWideClassifier

    all_results_dfs = []

    # Define scenarios
    # Moved to global SCENARIOS

    print(f"\nRunning Benchmark Scenarios A-H")

    if os.path.exists(output_file):
        res_df = pd.read_csv(output_file)
        # Ensure new columns exist
        if "masks_injected" not in res_df.columns:
            res_df["masks_injected"] = 0
        if "duplicate_factor" not in res_df.columns:
            res_df["duplicate_factor"] = 1
        if "scenario_name" not in res_df.columns:
            res_df["scenario_name"] = None
        if "n_estimators" not in res_df.columns:
            res_df["n_estimators"] = 8
    else:
        res_df = pd.DataFrame(columns=RESULT_COLUMNS)

    for scenario in SCENARIOS:
        name = scenario["name"]
        grp = scenario["grouping"]
        dup = scenario["dup"]
        mask = scenario["mask"]
        n_est = scenario["n_estimators"]
        ana_type = scenario["type"]

        print(f"\nRunning Scenario {name}: Grouping={grp}, Dup={dup}, Mask={mask}, N_est={n_est}")

        # Initialize classifier for this scenario
        clf = TabPFNWideClassifier(
            model_name="v2",
            device=device,
            n_estimators=n_est,
            ignore_pretraining_limits=True,
            features_per_group=grp,
        )

        for task_id in openml_df["tid"].values:
            task_id = int(task_id)

            # Check if already exists
            if not res_df.empty:
                cond = (
                    (res_df["task_id"] == task_id)
                    & (res_df["features_per_group"] == grp)
                    & (res_df["duplicate_factor"] == dup)
                    & (res_df["masks_injected"] == mask)
                    & (res_df["n_estimators"] == n_est)
                )
                if cond.any():
                    continue

            t = openml.tasks.get_task(task_id)
            res_df, _ = evaluate_task(
                task_id,
                grp,
                clf,
                device,
                t,
                res_df,
                duplicate_factor=dup,
                masks_injected=mask,
                n_estimators=n_est,
                analysis_type=ana_type,
                scenario_name=name,
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
            res_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
    all_results_dfs.append(res_df)

    # Combine all results for plotting and printing
    all_labels_order = None
    if all_results_dfs:
        final_df = pd.concat(all_results_dfs, ignore_index=True)
        print_all_results(final_df)

        if generate_plot:
            plot_output = output_file.replace(".csv", "_combined_plot.png")
            plot_combined_results(final_df, plot_output)

            plot_output = output_file.replace(".csv", "_combined_auroc.png")
            plot_combined_auroc_results(final_df, plot_output)

        # Get all unique labels for consistent coloring
        if "combo_label" not in final_df.columns:
            final_df["combo_label"] = final_df.apply(get_combo_label, axis=1)
        all_labels_order = sorted(final_df["combo_label"].unique())

    # --- Embedding Extraction and Analysis ---
    if extract_embeddings:
        print(f"\n=== Extracting Embeddings for Analysis ===")

        embedding_output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(embedding_output_dir, exist_ok=True)

        # Select subset of tasks for embedding analysis
        embedding_tasks = openml_df["tid"].values
        print(f"Analyzing embeddings for {len(embedding_tasks)} tasks")

        grouping_for_comparison = 2  # Use grouping=2 for all methods

        embeddings_dict = {}

        for task_id in embedding_tasks:
            task_id = int(task_id)
            t = openml.tasks.get_task(task_id)

            print(f"\n--- Extracting embeddings for task {task_id} ---")

            for scenario in SCENARIOS:
                name = scenario["name"]
                grp = scenario["grouping"]
                dup = scenario["dup"]
                mask = scenario["mask"]
                n_est = scenario["n_estimators"]
                ana_type = scenario["type"]

                print(f"  Method: {name} ({ana_type})")

                clf = TabPFNWideClassifier(
                    model_name="v2",
                    device=device,
                    n_estimators=n_est,
                    ignore_pretraining_limits=True,
                    features_per_group=grp,
                )

                _, emb = evaluate_task(
                    task_id,
                    grp,
                    clf,
                    device,
                    t,
                    pd.DataFrame(columns=RESULT_COLUMNS),
                    duplicate_factor=dup,
                    masks_injected=mask,
                    n_estimators=n_est,
                    extract_embeddings=True,
                    analysis_type=ana_type,
                    scenario_name=name,
                )

                if emb is not None:
                    key = f"t{task_id}_{name}"
                    embeddings_dict[key] = emb

        # Analyze
        if len(embeddings_dict) > 0:
            # Group by method type for visualization
            method_embeddings = {}

            # Initialize lists for all scenarios
            for scenario in SCENARIOS:
                label = get_combo_label(
                    {
                        "scenario_name": scenario["name"],
                        "analysis_type": scenario["type"],
                        "features_per_group": scenario["grouping"],
                        "duplicate_factor": scenario["dup"],
                        "masks_injected": scenario["mask"],
                        "n_estimators": scenario["n_estimators"],
                    }
                )
                method_embeddings[label] = []

            # Sort embeddings into the correct lists
            for key, emb in embeddings_dict.items():
                # key format: t{task_id}_{name}
                # Extract name (last part after underscore)
                scenario_name = key.split("_")[-1]

                # Find matching scenario
                matched_scenario = next((s for s in SCENARIOS if s["name"] == scenario_name), None)

                if matched_scenario:
                    label = get_combo_label(
                        {
                            "scenario_name": matched_scenario["name"],
                            "analysis_type": matched_scenario["type"],
                            "features_per_group": matched_scenario["grouping"],
                            "duplicate_factor": matched_scenario["dup"],
                            "masks_injected": matched_scenario["mask"],
                            "n_estimators": matched_scenario["n_estimators"],
                        }
                    )
                    method_embeddings[label].append(emb)

            # Concatenate embeddings for each method
            method_embeddings_concat = {}
            for method, emb_list in method_embeddings.items():
                if len(emb_list) > 0:
                    # Handle different shapes
                    try:
                        method_embeddings_concat[method] = np.concatenate(emb_list, axis=0)
                    except:
                        # If shapes differ, take the first one
                        method_embeddings_concat[method] = emb_list[0]

            analyze_embeddings(method_embeddings_concat, embedding_output_dir, all_labels_order)
        else:
            print("No embeddings were extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_id", type=int, help="OpenML suite ID")
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--min_features", type=int, default=100)
    parser.add_argument("--max_instances", type=int, default=10000)
    parser.add_argument(
        "--output_file", type=str, default="analysis_results/grouping_benchmark_results.csv"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--duplication_output_file",
        type=str,
        default="analysis_results/duplication_benchmark_results.csv",
    )
    parser.add_argument(
        "--masking_output_file", type=str, default="analysis_results/masking_benchmark_results.csv"
    )
    parser.add_argument(
        "--extract_embeddings", action="store_true", help="Extract and analyze embeddings"
    )
    parser.add_argument(
        "--tasks_limit",
        type=int,
        default=20,
        help="Number of tasks to analyze",
    )

    args = parser.parse_args()

    main(
        suite_id=args.suite_id,
        output_file=args.output_file,
        max_features=args.max_features,
        min_features=args.min_features,
        max_instances=args.max_instances,
        device=args.device,
        generate_plot=not args.no_plot,
        duplication_output_file=args.duplication_output_file,
        masking_output_file=args.masking_output_file,
        extract_embeddings=args.extract_embeddings,
        tasks_limit=args.tasks_limit,
    )
