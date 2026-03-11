import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import argparse
import scipy.stats as stats
import textwrap

# Set style for scientific plotting
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


MODEL_CONFIG = {
    "v2": {"color": "#1f77b4", "order": 0, "label": "TabPFN v2", "linestyle": ""},
    "wide-v2-1.5k": {
        "color": "#ff7f0e",
        "order": 1,
        "label": "TabPFN-Wide (1.5k)",
        "linestyle": "",
    },
    "wide-v2-1.5k-nocat": {
        "color": "#ffbb78",
        "order": 2,
        "label": "TabPFN-Wide (1.5k, No-Cat)",
        "linestyle": (2, 2),
    },
    "wide-v2-5k": {"color": "#2ca02c", "order": 3, "label": "TabPFN-Wide (5k)", "linestyle": ""},
    "wide-v2-5k-nocat": {
        "color": "#98df8a",
        "order": 4,
        "label": "TabPFN-Wide (5k, No-Cat)",
        "linestyle": (2, 2),
    },
    "wide-v2-8k": {"color": "#d62728", "order": 5, "label": "TabPFN-Wide (8k)", "linestyle": ""},
    "wide-v2-8k-nocat": {
        "color": "#ff9896",
        "order": 6,
        "label": "TabPFN-Wide (8k, No-Cat)",
        "linestyle": (2, 2),
    },
    "tabicl": {"color": "#9467bd", "order": 9, "label": "TabICL", "linestyle": (4, 2)},
    "random_forest": {
        "color": "#5D4037",
        "order": 10,
        "label": "Random Forest",
        "linestyle": (1, 1),
    },
    "xgboost": {
        "color": "#e377c2",
        "order": 11,
        "label": "XGBoost",
        "linestyle": (3, 1, 1, 1),
    },
    "realmlp": {
        "color": "#8c564b",
        "order": 12,
        "label": "RealMLP",
        "linestyle": (3, 1, 1, 1, 1, 1),
    },
}


def get_model_style(df, hue_col="checkpoint"):
    """
    Returns palette, order, and dashes for plotting based on MODEL_CONFIG.
    Also applies labels to the dataframe column in-place.

    Returns:
        palette: dict mapping model label to color
        order: list of model labels in display order
        dashes: dict mapping model label to linestyle tuple (or "" for solid)
    """
    if hue_col not in ["checkpoint", "model"]:
        # Fallback for non-model categorical plots
        unique_vals = df[hue_col].unique()
        return None, sorted(unique_vals), None

    # Apply labels to the column in-place
    label_map = {k: v["label"] for k, v in MODEL_CONFIG.items()}
    df[hue_col] = df[hue_col].apply(lambda x: label_map.get(x, x))

    models = df[hue_col].unique()
    label_to_config = {v["label"]: v for v in MODEL_CONFIG.values()}

    # Filter known models and sort by order
    known_models = sorted(
        [m for m in models if m in label_to_config],
        key=lambda m: label_to_config[m]["order"],
    )

    # Handle unknown models (append keeping alphabetical order)
    unknown_models = sorted([m for m in models if m not in label_to_config])

    # Combined order
    order = known_models + unknown_models

    # Create palette and dashes
    palette = {}
    dashes = {}
    for m in models:
        if m in label_to_config:
            palette[m] = label_to_config[m]["color"]
            dashes[m] = label_to_config[m].get("linestyle", "")
        else:
            palette[m] = "#333333"  # Dark Grey for unknown
            dashes[m] = ""  # Solid for unknown

    return palette, order, dashes


def clean_checkpoint_names(df, col="checkpoint"):
    """Clean checkpoint names by removing file extensions and paths."""
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).split("/")[-1].replace(".pt", ""))
    return df


def save_plots(fig, output_dir, filename_prefix):
    """Save figure as PDF and PNG."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{filename_prefix}.pdf")
    # png_path = os.path.join(output_dir, f"{filename_prefix}.png")

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    # fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved plots:\n  {pdf_path}")
    plt.close(fig)


def format_metric(metric):
    """Format metric name for display."""
    metric = metric.replace("_", " ").title()
    if metric == "Roc Auc":
        return "AUROC"
    return metric


def plot_categorical_comparison(
    df,
    x_col,
    y_col,
    hue_col=None,
    output_dir=None,
    basename=None,
    xlabel=None,
    ylabel=None,
    title=None,
    plot_type="box",
    ylim=(0.4, 1.05),
    suffix="",
    agg_threshold=40,
    figsize=(10, 6),
    xticks_rotation=45,
    wrap_width=None,
):
    """
    Generic plotting function for categorical comparisons.
    Automatically switches to aggregated view if x_col has too many categories.
    """
    if df.empty:
        return

    df = df.copy()

    unique_x = df[x_col].nunique()

    # Clean data
    df = df.dropna(subset=[y_col])
    if df.empty:
        print(f"Skipping plot for {y_col} - No valid data.")
        return
    if unique_x > agg_threshold:
        print(f"Comparison has {unique_x} categories for {x_col}, switching to aggregated view.")

        hue_order = None
        if hue_col:
            hue_order = df.groupby(hue_col)[y_col].median().sort_values(ascending=False).index

        plt.figure(figsize=figsize)
        sns.boxplot(
            data=df,
            x=hue_col,
            y=y_col,
            order=hue_order,
            palette="tab10",
            hue=hue_col,
            legend=False,
        )

        x_label_final = (
            xlabel if xlabel else (hue_col.replace("_", " ").title() if hue_col else x_col)
        )
        y_label_final = ylabel if ylabel else format_metric(y_col)

        plt.xlabel(x_label_final)
        plt.ylabel(y_label_final)

        plt.ylim(ylim)

        filename = f"{basename}_{y_col}_aggregated"
        if hue_col:
            filename += f"_by_{hue_col}"
        filename += suffix

        save_plots(plt.gcf(), output_dir, filename)
        return

    # Standard Plot
    plt.figure(figsize=figsize)

    # Seaborn 0.13.x has a bug when hue == x that causes UnboundLocalError.
    # Workaround: use hue=x with legend=False when we want colors on x.
    use_hue = hue_col and hue_col != x_col

    # Determine which column to use for colors
    color_col = hue_col if use_hue else (x_col if x_col in ["checkpoint", "model"] else None)

    # Get fixed palette and order if applicable
    # Note: get_model_style transforms the column values to labels in-place
    palette = "tab10"
    color_order = None
    if color_col in ["checkpoint", "model"]:
        palette, color_order, _ = get_model_style(df, color_col)

    # Sort x axis by median metric value (AFTER label transformation)
    order = df.groupby(x_col)[y_col].median().sort_values(ascending=False).index

    if plot_type == "box":
        if use_hue:
            sns.boxplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=hue_col,
                order=order,
                hue_order=color_order,
                palette=palette,
            )
        elif color_col:
            # Use hue=x for coloring, but disable legend (Seaborn 0.13 compatible)
            sns.boxplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=x_col,
                order=order,
                hue_order=color_order,
                palette=palette,
                legend=False,
            )
        else:
            sns.boxplot(
                data=df,
                x=x_col,
                y=y_col,
                order=order,
            )
    elif plot_type == "bar":
        if use_hue:
            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=hue_col,
                order=order,
                hue_order=color_order,
                palette=palette,
                errorbar="sd",
                capsize=0.1,
            )
        elif color_col:
            # Use hue=x for coloring, but disable legend (Seaborn 0.13 compatible)
            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=x_col,
                order=order,
                hue_order=color_order,
                palette=palette,
                errorbar="sd",
                capsize=0.1,
                legend=False,
            )
        else:
            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                order=order,
                errorbar="sd",
                capsize=0.1,
            )

    if wrap_width:
        locs, labels = plt.xticks()
        # If labels are empty (sometimes happens if drawn lazily), we might need to set them from order
        # But usually sns sets them.
        # Let's try to get current tick labels.
        current_labels = [l.get_text() for l in plt.gca().get_xticklabels()]
        if not current_labels and isinstance(order, pd.Index):
            current_labels = order.tolist()

        # If we still have labels
        if current_labels:
            wrapped_labels = [textwrap.fill(txt, width=wrap_width) for txt in current_labels]
            plt.xticks(
                plt.xticks()[0],
                wrapped_labels,
                rotation=xticks_rotation,
                ha="center" if xticks_rotation == 0 else "right",
            )
    else:
        plt.xticks(rotation=xticks_rotation, ha="center" if xticks_rotation == 0 else "right")

    plt.ylim(ylim)
    plt.xlabel(xlabel if xlabel else x_col.replace("_", " ").title())
    plt.ylabel(ylabel if ylabel else format_metric(y_col))

    # Handle legend: remove if not needed, reposition if needed
    ax = plt.gca()
    legend = ax.get_legend()
    if legend is not None:
        if use_hue and df[hue_col].nunique() > 1:
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(1.05, 1),
                title=hue_col.replace("_", " ").title(),
            )
        else:
            legend.remove()
    save_plots(plt.gcf(), output_dir, f"{basename}_{y_col}_per_{x_col}{suffix}")


def plot_line_comparison(
    df,
    x_col,
    y_col,
    hue_col=None,
    style_col=None,
    output_dir=None,
    basename=None,
    xlabel=None,
    ylabel=None,
    title=None,
    ylim=(0.4, 1.05),
    suffix="",
    smoothing=0,
):
    df = df.copy()
    if smoothing > 0:
        # Apply smoothing per group
        if hue_col:
            # Sort is crucial for rolling
            df = df.sort_values(by=[hue_col, x_col])
            df[y_col] = df.groupby(hue_col)[y_col].transform(
                lambda x: x.rolling(window=smoothing, min_periods=1, center=True).mean()
            )
        else:
            df = df.sort_values(by=x_col)
            df[y_col] = df[y_col].rolling(window=smoothing, min_periods=1, center=True).mean()

    # Filter out the dataset coverage problem
    if x_col == "n_features":
        # Check if we have data around 14000
        mask = (df[x_col] > 13000) & (df[x_col] < 14800)
        if mask.any():
            df = df[~mask]

    # Get fixed palette and order if applicable
    palette = "tab10"
    hue_order = None
    dashes = None
    if hue_col in ["checkpoint", "model"]:
        palette, hue_order, dashes = get_model_style(df, hue_col)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        hue_order=hue_order,
        style=style_col,
        style_order=hue_order if style_col == hue_col else None,
        markers=True,
        dashes=dashes if dashes else True,
        palette=palette,
        err_kws={"alpha": 0.05},
    )
    plt.ylim(ylim)
    plt.xlabel(xlabel if xlabel else x_col.replace("_", " ").title())
    plt.ylabel(ylabel if ylabel else format_metric(y_col))

    if hue_col:
        plt.legend(
            loc="best",
            title=hue_col.replace("_", " ").title(),
            frameon=False,
        )

    save_plots(plt.gcf(), output_dir, f"{basename}_{y_col}_vs_{x_col}{suffix}")


def plot_hdlss(df, output_dir, basename):
    """Plotting logic for HDLSS benchmarks."""
    output_dir = os.path.join(output_dir, "hdlss")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    for metric in metrics:
        # Plot 1: Per Dataset Comparison
        plot_categorical_comparison(
            df,
            x_col="dataset_name",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            xlabel="Dataset",
        )

        # Plot 2: Aggregated Summary (if allowed)
        if df["dataset_name"].nunique() > 1 and df["checkpoint"].nunique() > 1:
            plot_categorical_comparison(
                df,
                x_col="checkpoint",
                y_col=metric,
                output_dir=output_dir,
                basename=basename,
                plot_type="bar",
                # xlabel="Checkpoint",
                title=f"Aggregated {format_metric(metric)}",
                suffix="_overall_bar",
            )

        # Plot 3: One plot per dataset
        if df["checkpoint"].nunique() > 1:
            per_dataset_dir = os.path.join(output_dir, "per_dataset_plots")
            for ds in df["dataset_name"].unique():
                ds_df = df[df["dataset_name"] == ds]
                if ds_df.empty:
                    continue
                plot_categorical_comparison(
                    ds_df,
                    x_col="checkpoint",
                    y_col=metric,
                    hue_col="checkpoint",
                    output_dir=per_dataset_dir,
                    basename=f"{basename}_{ds}",
                    plot_type="bar",
                    # xlabel="Checkpoint",
                    title=f"{ds} - {format_metric(metric)}",
                    suffix="_comparison",
                )


def plot_reduced_hdlss_boxplot(df, output_dir, basename):
    """Compact reduced boxplot for HDLSS benchmarks (wide-5k + baselines only)."""
    output_dir = os.path.join(output_dir, "reduced_plots")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    # Keep only wide-5k and baselines
    keep_models = ["v2", "wide-v2-5k", "tabicl", "random_forest", "xgboost", "realmlp"]
    keep_labels = [MODEL_CONFIG[m]["label"] for m in keep_models if m in MODEL_CONFIG]
    df = df[df["checkpoint"].isin(keep_models + keep_labels)]

    if df.empty:
        return

    for metric in metrics:
        palette, hue_order, _ = get_model_style(df, "checkpoint")

        with sns.plotting_context("paper", font_scale=1.6):
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.boxplot(
                data=df,
                x="checkpoint",
                y=metric,
                hue="checkpoint",
                hue_order=hue_order,
                palette=palette,
                order=hue_order,
                ax=ax,
                width=0.6,
            )
            ax.set_ylim(0.4, 1.05)
            ax.set_xlabel("", fontsize=14)
            ax.set_ylabel(format_metric(metric), fontsize=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            plt.tight_layout()
            save_plots(fig, output_dir, f"{basename}_{metric}_boxplot_reduced")

        # Per-dataset comparison (all datasets in one plot)
        if "dataset_name" in df.columns and df["checkpoint"].nunique() > 1:
            for metric in metrics:
                plot_categorical_comparison(
                    df,
                    x_col="dataset_name",
                    y_col=metric,
                    hue_col="checkpoint",
                    output_dir=output_dir,
                    basename=basename,
                    xlabel="Dataset",
                    suffix=f"_{metric}_per_dataset_reduced",
                )


def plot_openml(df, output_dir, basename):
    """Plotting logic for OpenML benchmarks."""
    output_dir = os.path.join(output_dir, "openml")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    # Use task_id as categorical
    df["Task"] = "Task " + df["task_id"].astype(str)

    for metric in metrics:
        # Plot 1: Task vs Metric
        plot_categorical_comparison(
            df,
            x_col="Task",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
        )

        # Plot 2: Aggregated Summary
        if df["task_id"].nunique() > 1 and df["checkpoint"].nunique() > 1:
            plot_categorical_comparison(
                df,
                x_col="checkpoint",
                y_col=metric,
                output_dir=output_dir,
                basename=basename,
                plot_type="bar",
                # xlabel="Checkpoint",
                title=f"Aggregated {format_metric(metric)}",
                suffix="_overall_bar",
            )

        # Plot 3: One plot per task
        if df["checkpoint"].nunique() > 1:
            per_task_dir = os.path.join(output_dir, "per_task_plots")
            for task in df["task_id"].unique():
                task_df = df[df["task_id"] == task]
                if task_df.empty:
                    continue

                plot_categorical_comparison(
                    task_df,
                    x_col="checkpoint",
                    y_col=metric,
                    hue_col="checkpoint",
                    output_dir=per_task_dir,
                    basename=f"{basename}_task_{task}",
                    plot_type="bar",
                    # xlabel="Checkpoint",
                    title=f"Task {task} - {format_metric(metric)}",
                    suffix="_comparison",
                )


def plot_grouping(df, output_dir, basename):
    """Plotting logic for Grouping benchmarks."""
    output_dir = os.path.join(output_dir, "grouping")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]

    for metric in metrics:
        plot_categorical_comparison(
            df,
            x_col="features_per_group",
            y_col=metric,
            output_dir=output_dir,
            basename=basename,
            xlabel="Features Per Group",
            title=f"Impact of Grouping on {format_metric(metric)}",
            suffix="_grouping",
        )


def plot_multiomics(df, output_dir, basename):
    """Plotting logic for Multiomics Feature Reduction."""
    output_dir = os.path.join(output_dir, "multiomics")
    metrics = [c for c in ["accuracy", "roc_auc", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    for metric in metrics:
        # Separate plot per Dataset
        for ds in df["dataset_name"].unique():
            ds_df = df[df["dataset_name"] == ds]
            plot_line_comparison(
                ds_df,
                x_col="n_features",
                y_col=metric,
                hue_col="checkpoint",
                style_col="checkpoint",
                output_dir=output_dir,
                basename=basename,
                xlabel="Number of Features",
                title=f"{ds} - {format_metric(metric)} vs Feature Count",
                suffix=f"_{ds}_{metric}_feature_curve",
            )


def plot_multiomics_overview(df, output_dir, basename):
    """Multiomics overview plots."""
    output_dir = os.path.join(output_dir, "multiomics_overview")
    metrics = [c for c in ["accuracy", "roc_auc", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    # Filter to max 17500 features to exclude low coverage
    if "n_features" in df.columns:
        df = df[df["n_features"] <= 17500]

    for metric in metrics:
        # 1. Bar plot: Average per checkpoint
        # Aggregate logic
        df_agg = df.groupby("checkpoint")[metric].agg(["mean", "std"]).reset_index()
        df_agg = df_agg.sort_values("mean", ascending=False)

        # Custom logic for bar colors in overview
        palette, _, _ = get_model_style(df_agg, "checkpoint")
        bar_colors = [palette.get(x, "#333333") for x in df_agg["checkpoint"]]

        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(df_agg))
        plt.bar(
            x_pos,
            df_agg["mean"],
            yerr=df_agg["std"],
            capsize=5,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        plt.xticks(x_pos, df_agg["checkpoint"], rotation=45, ha="right")
        plt.ylim(0.45, 1.05)
        # plt.xlabel("Checkpoint")
        plt.ylabel(f"Average {format_metric(metric)}")

        plt.tight_layout()
        save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_overview_bar")

        # 2. Boxplot distribution
        plot_categorical_comparison(
            df,
            x_col="checkpoint",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            # xlabel="Checkpoint",
            title=f"Multiomics Overview - {format_metric(metric)} Distribution",
            suffix=f"_{metric}_overview_boxplot",
            figsize=(6, 6),
            xticks_rotation=0,
            wrap_width=10,
            ylim=(0.5, 1.0),
        )

        # 3. Line plot: Average metric vs n_features (Absolute)
        plot_line_comparison(
            df,
            x_col="n_features",
            y_col=metric,
            hue_col="checkpoint",
            style_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            xlabel="Number of Features",
            ylabel=f"Average {format_metric(metric)}",
            title=f"Multiomics Overview - {format_metric(metric)} vs Features",
            suffix=f"_{metric}_overview_features",
            # smoothing=5,  # Added smoothing
        )

        # 4. Relative Performance vs Baseline (v2)
        # We need to aggregate first to compute differences
        # Group by n_features and checkpoint
        df_agg = df.groupby(["n_features", "checkpoint"])[metric].mean().reset_index()

        baseline_name = "v2"
        baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
        unique_checkpoints = df_agg["checkpoint"].unique()
        actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

        if actual_baseline in unique_checkpoints:
            df_baseline = df_agg[df_agg["checkpoint"] == actual_baseline][["n_features", metric]]
            df_baseline = df_baseline.rename(columns={metric: "baseline_score"})

            df_merged = pd.merge(df_agg, df_baseline, on="n_features", how="left")
            df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]

            df_plot = df_merged[df_merged["checkpoint"] != actual_baseline]

            if not df_plot.empty:
                plot_line_comparison(
                    df_plot,
                    x_col="n_features",
                    y_col="relative_score",
                    hue_col="checkpoint",
                    style_col="checkpoint",
                    output_dir=output_dir,
                    basename=basename,
                    xlabel="Number of Features",
                    ylabel=f"Relative {format_metric(metric)} (vs {actual_baseline})",
                    title=f"Multiomics Overview - Relative {format_metric(metric)}",
                    suffix=f"_{metric}_overview_relative",
                    ylim=(-0.05, 0.4),  # Adjust ylim for relative plots
                    # smoothing=5,  # Added smoothing
                )


def plot_widening(df, output_dir, basename, comparison_mode=False):
    """Plotting logic for OpenML Widening."""
    output_dir = os.path.join(output_dir, "widening")
    metrics = [c for c in ["accuracy", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)

    has_sparsity = "sparsity" in df.columns
    sparsities = df["sparsity"].unique() if has_sparsity else [None]
    checkpoints = df["checkpoint"].unique() if "checkpoint" in df.columns else [None]
    datasets = df["dataset_id"].unique() if "dataset_id" in df.columns else [None]

    for metric in metrics:
        for ds in datasets:
            ds_df = df[df["dataset_id"] == ds] if ds is not None else df
            ds_label = f"Dataset {ds}" if ds is not None else basename
            ds_df = ds_df.sort_values("features_added")

            if comparison_mode and has_sparsity:
                # One plot per sparsity
                for sparsity in sparsities:
                    s_df = ds_df[ds_df["sparsity"] == sparsity]
                    if s_df.empty:
                        continue
                    plot_line_comparison(
                        s_df,
                        x_col="features_added",
                        y_col=metric,
                        hue_col="checkpoint",
                        style_col="checkpoint",
                        output_dir=output_dir,
                        basename=basename,
                        title=f"{ds_label} - {metric} (Sparsity={sparsity})",
                        xlabel="Features Added",
                        suffix=f"_{ds}_{metric}_sparsity_{sparsity}_comparison",
                    )

            elif has_sparsity and len(sparsities) > 1:
                # Single model, multiple sparsities
                ds_df["sparsity_label"] = ds_df["sparsity"].apply(lambda x: f"Sparsity={x}")
                plot_line_comparison(
                    ds_df,
                    x_col="features_added",
                    y_col=metric,
                    hue_col="sparsity_label",
                    style_col="sparsity_label",
                    output_dir=output_dir,
                    basename=basename,
                    title=f"{ds_label} - {metric} vs Features Added",
                    xlabel="Features Added",
                    suffix=f"_{ds}_{metric}_all_sparsities",
                )

                # Also per sparsity
                per_sparsity_dir = os.path.join(output_dir, "per_sparsity")
                for sparsity in sparsities:
                    s_df = ds_df[ds_df["sparsity"] == sparsity]
                    if s_df.empty:
                        continue
                    plot_line_comparison(
                        s_df,
                        x_col="features_added",
                        y_col=metric,
                        hue_col="checkpoint" if len(checkpoints) > 1 else None,
                        style_col="checkpoint" if len(checkpoints) > 1 else None,
                        output_dir=per_sparsity_dir,
                        basename=basename,
                        title=f"{ds_label} - {metric} (Sparsity={sparsity})",
                        xlabel="Features Added",
                        suffix=f"_{ds}_{metric}_sparsity_{sparsity}",
                    )
            else:
                # Default
                plot_line_comparison(
                    ds_df,
                    x_col="features_added",
                    y_col=metric,
                    hue_col="checkpoint" if len(checkpoints) > 1 else None,
                    style_col="checkpoint" if len(checkpoints) > 1 else None,
                    output_dir=output_dir,
                    basename=basename,
                    title=f"{ds_label} - {metric} vs Features Added",
                    xlabel="Features Added",
                    suffix=f"_{ds}_{metric}_widening_curve",
                )


def plot_widening_relative(df, output_dir, basename, baseline_name="v2"):
    """
    Plot average AUROC relative to a baseline model.
    """
    output_dir = os.path.join(output_dir, "widening")
    metric = "roc_auc_score"
    if metric not in df.columns:
        return
    if basename == baseline_name:
        return

    df = clean_checkpoint_names(df)

    if "sparsity" not in df.columns or "features" not in df.columns:
        print("Missing 'sparsity' or 'features' columns for relative plots.")
        return

    # Aggregate
    group_cols = ["checkpoint", "sparsity", "features"]
    df_agg = df.groupby(group_cols)[metric].mean().reset_index()

    # 1. Absolute Average Performance
    unique_sparsities = sorted(df_agg["sparsity"].unique())
    for sp in unique_sparsities:
        sp_df = df_agg[df_agg["sparsity"] == sp].sort_values("features")
        if sp_df.empty:
            continue

        plot_line_comparison(
            sp_df,
            x_col="features",
            y_col=metric,
            hue_col="checkpoint",
            style_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            title=f"Average Performance (All Datasets) - Sparsity {sp}",
            xlabel="Total Features",
            ylabel=f"Average {format_metric(metric)}",
            suffix=f"_average_auc_absolute_sparsity_{sp}",
        )

    # 2. Relative Performance
    baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
    unique_checkpoints = df_agg["checkpoint"].unique()
    actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

    df_baseline = df_agg[df_agg["checkpoint"] == actual_baseline].copy()
    if df_baseline.empty:
        print(f"Baseline '{actual_baseline}' not found for relative plots.")
        return

    df_baseline = df_baseline.rename(columns={metric: "baseline_score"}).drop(
        columns=["checkpoint"]
    )
    df_merged = pd.merge(df_agg, df_baseline, on=["sparsity", "features"], how="left")
    df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]
    df_merged = df_merged.dropna(subset=["relative_score"])

    for sp in unique_sparsities:
        sp_df = df_merged[df_merged["sparsity"] == sp].sort_values("features")
        sp_df = sp_df[sp_df["checkpoint"] != actual_baseline].copy()
        if sp_df.empty:
            continue

        palette, hue_order, dashes = get_model_style(sp_df, "checkpoint")

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=sp_df,
            x="features",
            y="relative_score",
            hue="checkpoint",
            hue_order=hue_order,
            style="checkpoint",
            style_order=hue_order,
            markers=True,
            dashes=dashes if dashes else True,
            palette=palette,
        )
        plt.ylim(-0.1, 0.35)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)

        plt.xlabel("Total Features")
        plt.ylabel(f"Relative AUROC (vs {actual_baseline})")
        plt.legend(loc="best", frameon=False)
        save_plots(plt.gcf(), output_dir, f"{basename}_relative_auc_sparsity_{sp}")


def plot_forgetting(df, output_dir, basename):
    """Scatter plot comparing two models (Forgetting)."""
    output_dir = os.path.join(output_dir, "forgetting")

    required_cols = ["task_id", "checkpoint", "roc_auc_score"]
    if not all(col in df.columns for col in required_cols):
        return

    df = df.copy()

    # Filter
    # Filter: include v2 and wide models (supporting both raw and labeled names)
    def is_target(x):
        x_str = str(x)
        return (
            x_str == "v2"
            or x_str == "TabPFN v2"
            or x_str.startswith("wide-v2")
            or x_str.startswith("Wide")
        )

    df = df[df["checkpoint"].apply(is_target)]
    if df.empty:
        return

    df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")
    df = df.dropna(subset=["task_id"])

    # Aggregate
    df_agg = df.groupby(["task_id", "checkpoint"])["roc_auc_score"].mean().reset_index()
    df_pivot = df_agg.pivot(index="task_id", columns="checkpoint", values="roc_auc_score")

    if df_pivot.shape[1] < 2:
        return

    cols = df_pivot.columns
    baseline = cols[0]
    others = cols[1:]

    for other in others:
        df_xy = df_pivot[[baseline, other]].dropna()
        if df_xy.empty:
            continue

        x_vals = df_xy[baseline]
        y_vals = df_xy[other]
        rho, _ = stats.spearmanr(x_vals, y_vals)

        plt.figure(figsize=(6, 6))
        plt.scatter(x_vals, y_vals, marker="x", s=100, linewidths=1.5, color="mediumblue")
        plt.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", linewidth=1.5)

        plt.xlim(0.5, 1.0)
        plt.ylim(0.5, 1.0)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(f"{baseline}", fontsize=12)
        plt.ylabel(f"{other}", fontsize=12)
        plt.title(f"Forgetting Comparison (Spearman rho={rho:.4f})")
        save_plots(plt.gcf(), output_dir, f"forgetting_scatter_{baseline}_vs_{other}")


def plot_snp(df, output_dir, basename):
    """Plotting logic for SNP benchmarks."""
    output_dir = os.path.join(output_dir, "snp")
    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])

    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    df = clean_checkpoint_names(df, col=hue_col)

    metrics = [c for c in ["roc_auc", "accuracy"] if c in df.columns]

    for metric in metrics:
        if "Polygenicity" in df.columns:
            # Check if we are in single-model mode (non-comparison)
            if df[hue_col].nunique() == 1:
                # One plot with all Polygenicities
                model_name = df[hue_col].iloc[0]

                # Convert Polygenicity to string for categorical hue
                df["Polygenicity_Label"] = df["Polygenicity"].astype(str)
                poly_order = sorted(df["Polygenicity"].unique())
                poly_order_str = [str(x) for x in poly_order]

                plt.figure(figsize=(10, 6))
                sns.lineplot(
                    data=df,
                    x="n_features",
                    y=metric,
                    hue="Polygenicity_Label",
                    hue_order=poly_order_str,
                    style="Polygenicity_Label",
                    style_order=poly_order_str,
                    markers=True,
                    dashes=False,  # Solid lines for clarity
                    palette="tab10",  # Match standard categorical palette
                    err_kws={"alpha": 0.05},
                )
                plt.ylim(0.4, 1)
                plt.xlabel("Number of Features")
                plt.ylabel(format_metric(metric))
                plt.title(f"{model_name} - {format_metric(metric)} across Polygenicities")
                plt.legend(
                    loc="best",
                    title="Polygenicity",
                    frameon=False,
                )
                save_plots(plt.gcf(), output_dir, f"{basename}_{metric}_all_polygenicities")

            else:
                # Comparison mode: One plot per Polygenicity
                palette, hue_order, dashes = get_model_style(df, hue_col)

                for polygenicity in sorted(df["Polygenicity"].unique()):
                    poly_df = df[df["Polygenicity"] == polygenicity]
                    if poly_df.empty:
                        continue

                    plt.figure(figsize=(10, 6))
                    sns.lineplot(
                        data=poly_df,
                        x="n_features",
                        y=metric,
                        hue=hue_col,
                        hue_order=hue_order,
                        style=hue_col,
                        style_order=hue_order,
                        markers=True,
                        dashes=dashes if dashes else True,
                        palette=palette,
                        err_kws={"alpha": 0.05},
                    )
                    plt.ylim(0.4, 1)
                    plt.xlabel("Number of Features")
                    plt.ylabel(format_metric(metric))
                    # plt.title(f"Polygenicity: {polygenicity}")
                    plt.legend(
                        loc="best",
                        title=hue_col.replace("_", " ").title(),
                        frameon=False,
                    )
                    save_plots(
                        plt.gcf(), output_dir, f"{basename}_{metric}_polygenicity_{polygenicity}"
                    )
        else:
            plot_line_comparison(
                df,
                x_col="n_features",
                y_col=metric,
                hue_col=hue_col,
                style_col=hue_col,
                output_dir=output_dir,
                basename=basename,
                xlabel="Number of Features",
                suffix=f"_{metric}",
            )


# Models to include in reduced plots (using raw checkpoint names)
REDUCED_MODELS = [
    "v2",
    "wide-v2-1.5k",
    "wide-v2-5k",
    "wide-v2-8k",
    "tabicl",
    "random_forest",
    "xgboost",
    "realmlp",
]
REDUCED_MODELS_LABELS = [
    "TabPFNv2",
    "TabPFN-Wide (1.5k)",
    "TabPFN-Wide (5k)",
    "TabPFN-Wide (8k)",
    "TabICL",
    "Random Forest",
    "XGBoost",
    "RealMLP",
]


def filter_to_reduced_models(df, col="checkpoint"):
    """Filter dataframe to only include reduced model set."""
    # Filter by raw names or labels
    mask = df[col].isin(REDUCED_MODELS) | df[col].isin(REDUCED_MODELS_LABELS)
    return df[mask].copy()


def plot_reduced_widening_relative(df, output_dir, basename, baseline_name="v2"):
    """Reduced plot for widening relative performance."""
    output_dir = os.path.join(output_dir, "reduced_plots")
    metric = "roc_auc_score"
    if metric not in df.columns:
        return

    df = clean_checkpoint_names(df)
    df = filter_to_reduced_models(df)

    if df.empty:
        print("No data after filtering for reduced models (widening relative).")
        return

    if "sparsity" not in df.columns or "features" not in df.columns:
        return

    # Aggregate
    group_cols = ["checkpoint", "sparsity", "features"]
    df_agg = df.groupby(group_cols)[metric].mean().reset_index()

    # Relative Performance
    baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
    unique_checkpoints = df_agg["checkpoint"].unique()
    actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

    df_baseline = df_agg[df_agg["checkpoint"] == actual_baseline].copy()
    if df_baseline.empty:
        return

    df_baseline = df_baseline.rename(columns={metric: "baseline_score"}).drop(
        columns=["checkpoint"]
    )
    df_merged = pd.merge(df_agg, df_baseline, on=["sparsity", "features"], how="left")
    df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]
    df_merged = df_merged.dropna(subset=["relative_score"])

    # Only plot sparsity 0.25
    sp = 0.25
    sp_df = df_merged[df_merged["sparsity"] == sp].sort_values("features")
    sp_df = sp_df[sp_df["checkpoint"] != actual_baseline].copy()
    if sp_df.empty:
        return

    palette, hue_order, dashes = get_model_style(sp_df, "checkpoint")

    # Publication styling
    with sns.plotting_context("paper", font_scale=1.6):
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.lineplot(
            data=sp_df,
            x="features",
            y="relative_score",
            hue="checkpoint",
            hue_order=hue_order,
            style="checkpoint",
            style_order=hue_order,
            markers=True,
            dashes=dashes if dashes else True,
            palette=palette,
            linewidth=2.5,
            markersize=8,
            errorbar="sd",
            err_kws={"alpha": 0.1},
            ax=ax,
        )
        ax.set_ylim(-0.05, 0.25)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Total Features", fontsize=14)
        ax.set_ylabel(f"Relative AUROC (vs {actual_baseline})", fontsize=14)
        ax.legend(loc="best", fontsize=11, frameon=False)
        plt.tight_layout()
        save_plots(fig, output_dir, f"{basename}_relative_auc_sparsity_{sp}_reduced")


def plot_reduced_snp(df, output_dir, basename):
    """Reduced plot for SNP benchmark."""
    output_dir = os.path.join(output_dir, "reduced_plots")
    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])

    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    df = clean_checkpoint_names(df, col=hue_col)
    df = filter_to_reduced_models(df, col=hue_col)
    # Exclude 1.5k and 8k variants from this plot
    exclude = ["wide-v2-1.5k", "wide-v2-8k", "TabPFN-Wide (1.5k)", "TabPFN-Wide (8k)"]
    df = df[~df[hue_col].isin(exclude)]

    if df.empty:
        print("No data after filtering for reduced models (SNP).")
        return

    metric = "roc_auc"
    if metric not in df.columns:
        return

    if "Polygenicity" not in df.columns:
        return

    # Plot for each polygenicity
    for polygenicity in sorted(df["Polygenicity"].unique()):
        poly_df = df[df["Polygenicity"] == polygenicity]
        if poly_df.empty:
            continue

        palette, hue_order, dashes = get_model_style(poly_df, hue_col)

        # Publication styling
        with sns.plotting_context("paper", font_scale=1.6):
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.lineplot(
                data=poly_df,
                x="n_features",
                y=metric,
                hue=hue_col,
                hue_order=hue_order,
                style=hue_col,
                style_order=hue_order,
                markers=True,
                dashes=dashes if dashes else True,
                palette=palette,
                linewidth=2.5,
                markersize=8,
                err_kws={"alpha": 0.1},
                ax=ax,
            )
            ax.set_ylim(0.35, 1)
            ax.set_xlabel("Number of Features", fontsize=14)
            ax.set_ylabel(format_metric(metric), fontsize=14)
            ax.legend(loc="lower right", fontsize=11, frameon=False)
            plt.tight_layout()
            save_plots(fig, output_dir, f"{basename}_{metric}_polygenicity_{polygenicity}_reduced")


def plot_snp_relative(df, output_dir, basename, baseline_name="v2"):
    """
    Plot SNP AUROC relative to a baseline model.
    """
    output_dir = os.path.join(output_dir, "snp")
    metric = "roc_auc"
    if metric not in df.columns:
        return
    if "Polygenicity" not in df.columns:
        return

    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])

    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    df = clean_checkpoint_names(df, col=hue_col)

    # Aggregate
    group_cols = [hue_col, "Polygenicity", "n_features"]
    df_agg = df.groupby(group_cols)[metric].mean().reset_index()

    # Get baseline
    baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
    unique_checkpoints = df_agg[hue_col].unique()
    actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

    df_baseline = df_agg[df_agg[hue_col] == actual_baseline].copy()
    if df_baseline.empty:
        return

    df_baseline = df_baseline.rename(columns={metric: "baseline_score"}).drop(columns=[hue_col])

    df_merged = pd.merge(df_agg, df_baseline, on=["Polygenicity", "n_features"], how="left")
    df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]

    # Plot for each Polygenicity
    for poly in df_merged["Polygenicity"].unique():
        poly_df = df_merged[df_merged["Polygenicity"] == poly]
        # remove baseline from plot
        poly_df = poly_df[poly_df[hue_col] != actual_baseline]

        if poly_df.empty:
            continue

        plot_line_comparison(
            poly_df,
            x_col="n_features",
            y_col="relative_score",
            hue_col=hue_col,
            style_col=hue_col,
            output_dir=output_dir,
            basename=basename,
            xlabel="Number of Features",
            ylabel=f"Relative {format_metric(metric)} (vs {actual_baseline})",
            title=f"Relative {format_metric(metric)} - Polygenicity {poly}",
            suffix=f"_{metric}_polygenicity_{poly}_relative",
            ylim=(-0.2, 0.35),
        )


def plot_reduced_snp_relative(df, output_dir, basename, baseline_name="v2"):
    """Reduced plot for SNP relative performance."""
    output_dir = os.path.join(output_dir, "reduced_plots")

    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])

    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    df = clean_checkpoint_names(df, col=hue_col)
    df = filter_to_reduced_models(df, col=hue_col)
    # Exclude 1.5k and 8k variants from this plot
    exclude = ["wide-v2-1.5k", "wide-v2-8k", "TabPFN-Wide (1.5k)", "TabPFN-Wide (8k)"]
    df = df[~df[hue_col].isin(exclude)]

    if df.empty:
        return

    metric = "roc_auc"
    if metric not in df.columns:
        return
    if "Polygenicity" not in df.columns:
        return

    # Plot for each polygenicity
    for polygenicity in sorted(df["Polygenicity"].unique()):
        df_poly = df[df["Polygenicity"] == polygenicity]
        if df_poly.empty:
            continue

        # Aggregate
        group_cols = [hue_col, "n_features"]
        df_agg = df_poly.groupby(group_cols)[metric].mean().reset_index()

        # Calculate Relative
        baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
        unique_checkpoints = df_agg[hue_col].unique()
        actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

        df_baseline = df_agg[df_agg[hue_col] == actual_baseline].copy()
        if df_baseline.empty:
            continue

        df_baseline = df_baseline.rename(columns={metric: "baseline_score"}).drop(columns=[hue_col])

        df_merged = pd.merge(df_agg, df_baseline, on=["n_features"], how="left")
        df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]

        # Remove baseline from plot
        df_plot = df_merged[df_merged[hue_col] != actual_baseline].copy()
        if df_plot.empty:
            continue

        palette, hue_order, dashes = get_model_style(df_plot, hue_col)

        with sns.plotting_context("paper", font_scale=1.6):
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.lineplot(
                data=df_plot,
                x="n_features",
                y="relative_score",
                hue=hue_col,
                hue_order=hue_order,
                style=hue_col,
                style_order=hue_order,
                markers=True,
                dashes=dashes if dashes else True,
                palette=palette,
                linewidth=2.5,
                markersize=8,
                ax=ax,
                errorbar="sd",
                err_kws={"alpha": 0.1},
            )
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            # ax.set_ylim(-0.05, 0.25)
            ax.set_xlabel("Number of Features", fontsize=14)
            ax.set_ylabel(f"Relative {format_metric(metric)} (vs {actual_baseline})", fontsize=14)
            ax.legend(loc="lower right", fontsize=11, frameon=False)
            plt.tight_layout()
            save_plots(
                fig, output_dir, f"{basename}_{metric}_polygenicity_{polygenicity}_reduced_relative"
            )


def plot_reduced_snp_relative_polygenicity_comparison(df, output_dir, basename, baseline_name="v2"):
    """
    Reduced plot for SNP relative performance: One plot per model vs v2 (baseline) over all polygenicities.
    Generates a separate plot for every model found in the dataframe (excluding the baseline).
    """
    output_dir = os.path.join(output_dir, "reduced_plots")

    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])

    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    df = clean_checkpoint_names(df, col=hue_col)

    metric = "roc_auc"
    if metric not in df.columns:
        return
    if "Polygenicity" not in df.columns:
        return

    # Aggregate
    group_cols = [hue_col, "Polygenicity", "n_features"]
    df_agg = df.groupby(group_cols)[metric].mean().reset_index()

    # Get baseline (v2 without smearing -> initial performance at min features)
    baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
    unique_checkpoints = df_agg[hue_col].unique()
    actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

    df_baseline = df_agg[df_agg[hue_col] == actual_baseline].copy()
    if df_baseline.empty:
        return

    df_baseline = df_baseline.rename(columns={metric: "baseline_score"}).drop(columns=[hue_col])

    # Merge on Polygenicity and n_features (matched feature counts)
    df_merged = pd.merge(df_agg, df_baseline, on=["Polygenicity", "n_features"], how="left")
    df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]

    # Iterate over all models except baseline
    unique_models = [m for m in df_merged[hue_col].unique() if m != actual_baseline]

    for model_name in unique_models:
        df_plot = df_merged[df_merged[hue_col] == model_name].copy()
        if df_plot.empty:
            continue

        # Get config info
        config = MODEL_CONFIG.get(model_name.removesuffix("-nocat"), {})
        model_color = config.get("color", "#333333")
        model_label = config.get("label", model_name)

        # Convert Polygenicity to categorical for proper hue ordering/shading
        df_plot["Polygenicity"] = df_plot["Polygenicity"].astype(str)
        poly_order = sorted(df_plot["Polygenicity"].unique(), key=lambda x: float(x))

        # Generate Palette: gradient based on model color
        # Skip the first few very light colors to ensure visibility
        try:
            palette = sns.light_palette(model_color, n_colors=len(poly_order) + 3)[3:]
        except Exception:
            palette = "viridis"  # Fallback

        safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()

        with sns.plotting_context("paper", font_scale=1.6):
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.lineplot(
                data=df_plot,
                x="n_features",
                y="relative_score",
                hue="Polygenicity",
                hue_order=poly_order,
                style="Polygenicity",
                style_order=poly_order,
                markers=True,
                dashes=False,
                palette=palette,
                linewidth=2.5,
                markersize=8,
                ax=ax,
                errorbar="sd",
                err_kws={"alpha": 0.1},
            )
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Number of Features", fontsize=14)
            ax.set_ylabel(f"Relative {format_metric(metric)} (vs {actual_baseline})", fontsize=14)
            ax.legend(loc="best", fontsize=11, frameon=False, title="Polygenicity")
            # ax.set_title(f"{model_label}", fontsize=16)
            ax.set_ylim((-0.2, 0.2))
            plt.tight_layout()
            save_plots(
                fig,
                output_dir,
                f"{basename}_{metric}_{safe_model_name}_reduced_relative_polygenicity_comparison",
            )


def plot_reduced_snp_average_poly(df, output_dir, basename):
    """Reduced plot for SNP benchmark averaging over all polygenicities."""
    output_dir = os.path.join(output_dir, "reduced_plots")
    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"])

    hue_col = "checkpoint" if "checkpoint" in df.columns else "model"
    df = clean_checkpoint_names(df, col=hue_col)
    df = filter_to_reduced_models(df, col=hue_col)
    # Exclude 1.5k and 8k variants from this plot
    exclude = ["wide-v2-1.5k", "wide-v2-8k", "TabPFN-Wide (1.5k)", "TabPFN-Wide (8k)"]
    df = df[~df[hue_col].isin(exclude)]

    if df.empty:
        return

    metric = "roc_auc"
    if metric not in df.columns:
        return

    if "Polygenicity" not in df.columns:
        return

    palette, hue_order, dashes = get_model_style(df, hue_col)

    with sns.plotting_context("paper", font_scale=1.6):
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.lineplot(
            data=df,
            x="n_features",
            y=metric,
            hue=hue_col,
            hue_order=hue_order,
            style=hue_col,
            style_order=hue_order,
            markers=True,
            dashes=dashes if dashes else True,
            palette=palette,
            linewidth=2.5,
            markersize=8,
            errorbar="sd",
            err_kws={"alpha": 0.1},
            ax=ax,
        )
        ax.set_ylim(0.4, 1)
        ax.set_xlabel("Number of Features", fontsize=14)
        ax.set_ylabel(f"Average {format_metric(metric)}", fontsize=14)
        ax.legend(loc="best", fontsize=11, frameon=False)
        plt.tight_layout()
        save_plots(fig, output_dir, f"{basename}_{metric}_average_poly_reduced")


def plot_reduced_multiomics_overview(df, output_dir, basename):
    """Reduced plot for Multiomics Overview (Absolute)."""
    output_dir = os.path.join(output_dir, "reduced_plots")
    metrics = [c for c in ["roc_auc", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)
    df = filter_to_reduced_models(df)
    # Exclude 1.5k and 8k variants from this plot
    exclude = ["wide-v2-1.5k", "wide-v2-8k", "TabPFN-Wide (1.5k)", "TabPFN-Wide (8k)"]
    df = df[~df["checkpoint"].isin(exclude)]

    if df.empty:
        return

    # Filter to max 17500 features
    if "n_features" in df.columns:
        df = df[df["n_features"] <= 17500]
        # Filter out the dataset coverage problem
        mask = (df["n_features"] > 13000) & (df["n_features"] < 14800)
        if mask.any():
            df = df[~mask]

    for metric in metrics:
        palette, hue_order, dashes = get_model_style(df, "checkpoint")

        # Line plot: Average metric vs n_features (Absolute)
        with sns.plotting_context("paper", font_scale=1.6):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.lineplot(
                data=df,
                x="n_features",
                y=metric,
                hue="checkpoint",
                hue_order=hue_order,
                style="checkpoint",
                style_order=hue_order,
                markers=True,
                dashes=dashes if dashes else True,
                palette=palette,
                linewidth=2.5,
                markersize=8,
                errorbar="sd",
                err_kws={"alpha": 0.1},
                ax=ax,
            )
            ax.set_ylim(0.45, 1.05)
            ax.set_xlabel("Number of Features", fontsize=14)
            ax.set_ylabel(f"Average {format_metric(metric)}", fontsize=14)
            ax.legend(loc="best", fontsize=11, frameon=False)
            plt.tight_layout()
            save_plots(fig, output_dir, f"{basename}_{metric}_overview_features_reduced")


def plot_reduced_multiomics_overview_boxplot(df, output_dir, basename):
    """
    Reduced Boxplot distribution (Wide only, no-cat excluded).
    Saves to 'reduced_plots' subdirectory.
    """
    output_dir = os.path.join(output_dir, "reduced_plots")
    metrics = [c for c in ["roc_auc", "roc_auc_score"] if c in df.columns]

    # Filter to max 17500 features (same as overview)
    if "n_features" in df.columns:
        df = df[df["n_features"] <= 17500]

    # Filter logic: Wide models only (exclude nocat) + v2 baseline
    df_reduced = df[
        (df["checkpoint"] == "v2")
        | (
            df["checkpoint"].str.contains("wide", case=False)
            & ~df["checkpoint"].str.contains("nocat", case=False)
        )
    ].copy()

    if df_reduced.empty:
        return

    for metric in metrics:
        plot_categorical_comparison(
            df_reduced,
            x_col="checkpoint",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            xlabel="",
            title=f"Multiomics Overview (Reduced) - {format_metric(metric)} Distribution",
            suffix=f"_{metric}_overview_boxplot_reduced",
            figsize=(6, 6),
            xticks_rotation=45,
            wrap_width=None,
            ylim=(0.5, 1.0),
        )


def plot_reduced_multiomics_overview_boxplot_with_baselines(df, output_dir, basename):
    """
    Reduced Boxplot distribution including baselines.
    Saves to 'reduced_plots' subdirectory.
    """
    output_dir = os.path.join(output_dir, "reduced_plots")
    metrics = [c for c in ["roc_auc", "roc_auc_score"] if c in df.columns]

    # Filter to max 17500 features (same as overview)
    if "n_features" in df.columns:
        df = df[df["n_features"] <= 17500]

    df_reduced = clean_checkpoint_names(df)
    df_reduced = filter_to_reduced_models(df_reduced)

    if df_reduced.empty:
        return

    for metric in metrics:
        plot_categorical_comparison(
            df_reduced,
            x_col="checkpoint",
            y_col=metric,
            hue_col="checkpoint",
            output_dir=output_dir,
            basename=basename,
            xlabel="",
            title=f"Multiomics Overview (Reduced) - {format_metric(metric)} Distribution",
            suffix=f"_{metric}_overview_boxplot_reduced_with_baselines",
            figsize=(6, 6),
            xticks_rotation=45,
            wrap_width=None,
            ylim=(0.5, 1.0),
        )


def _generate_multiomics_latex_table(df, output_dir, basename, metric, metric_display_name):
    """
    Generate a LaTeX table for Multiomics metrics (Dataset columns x Model rows).

    Args:
        df: DataFrame with multiomics results
        output_dir: Output directory for the table
        basename: Base name for the output file
        metric: Column name for the metric (e.g., "roc_auc", "accuracy")
        metric_display_name: Display name for the metric in caption (e.g., "AUROC", "Accuracy")
    """
    os.makedirs(output_dir, exist_ok=True)

    if metric not in df.columns:
        print(f"Metric '{metric}' not found in columns for LaTeX table.")
        return

    df = df.copy()
    if "n_features" in df.columns:
        df["n_features"] = pd.to_numeric(df["n_features"], errors="coerce")

    df = clean_checkpoint_names(df)

    # Label map for display
    label_map = {k: v["label"] for k, v in MODEL_CONFIG.items()}

    def get_display_label(name):
        if name in label_map:
            return label_map[name]
        if name in label_map.values():
            return name
        name_lower = str(name).lower()
        if name_lower in label_map:
            return label_map[name_lower]
        return None

    df["display_label"] = df["checkpoint"].apply(get_display_label)
    df = df.dropna(subset=["display_label"])
    df["checkpoint"] = df["display_label"]

    target_models_order = [
        "TabPFN v2",
        "TabPFN-Wide (1.5k)",
        "TabPFN-Wide (1.5k, No-Cat)",
        "TabPFN-Wide (5k)",
        "TabPFN-Wide (5k, No-Cat)",
        "TabPFN-Wide (8k)",
        "TabPFN-Wide (8k, No-Cat)",
        "TabICL",
        "Random Forest",
        "XGBoost",
        "RealMLP",
    ]

    df = df[df["checkpoint"].isin(target_models_order)]

    if df.empty:
        print(f"No matching models found for {metric_display_name} LaTeX table.")
        return

    # For each (dataset, checkpoint), select rows at maximum n_features
    if "n_features" in df.columns:
        max_feat = df.groupby(["dataset_name", "checkpoint"])["n_features"].transform("max")
        df = df[df["n_features"] == max_feat]

    # Aggregate mean/std over folds
    df_agg = df.groupby(["dataset_name", "checkpoint"])[metric].agg(["mean", "std"]).reset_index()

    # Pivot: Dataset columns, Model rows
    pivot_mean = df_agg.pivot(index="checkpoint", columns="dataset_name", values="mean")
    pivot_std = df_agg.pivot(index="checkpoint", columns="dataset_name", values="std")

    # Reorder rows according to target order
    pivot_mean = pivot_mean.reindex(target_models_order).dropna(how="all")
    pivot_std = pivot_std.reindex(target_models_order).loc[pivot_mean.index]

    if pivot_mean.empty:
        print(f"No data for {metric_display_name} LaTeX table after aggregation.")
        return

    col_names = pivot_mean.columns.tolist()

    # Get max n_features per dataset for header
    n_feats = []
    for ds in col_names:
        feat_val = df[df["dataset_name"] == ds]["n_features"].max()
        n_feats.append(f"{int(feat_val):,}" if pd.notnull(feat_val) else "-")

    # Build LaTeX
    latex_str = (
        "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l"
        + "c" * len(col_names)
        + "}\n\\toprule\n"
    )
    latex_str += "Model & " + " & ".join(col_names) + " \\\\\n"
    latex_str += "\\#features & " + " & ".join(n_feats) + " \\\\\n"
    latex_str += "\\midrule\n"

    for model in pivot_mean.index:
        row_str = f"{model} "
        for ds in col_names:
            mean = pivot_mean.loc[model, ds]
            std = pivot_std.loc[model, ds]

            if pd.isna(mean):
                row_str += "& - "
                continue

            best_mean = pivot_mean[ds].max()
            is_best = abs(mean - best_mean) < 1e-4

            val_str = f"${mean:.3f} \\pm {std:.3f}$" if pd.notna(std) else f"${mean:.3f}$"
            if is_best:
                val_str = (
                    f"$\\textbf{{{mean:.3f} \\pm {std:.3f}}}$"
                    if pd.notna(std)
                    else f"$\\textbf{{{mean:.3f}}}$"
                )

            row_str += f"& {val_str} "

        row_str += "\\\\\n"
        latex_str += row_str

        if model == "TabPFN-Wide (8k, No-Cat)":
            latex_str += "\\midrule\n"

    metric_label = metric_display_name.lower().replace(" ", "_")
    latex_str += f"\\bottomrule\n\\end{{tabular}}\n\\caption{{Multiomics {metric_display_name} Comparison}}\n\\label{{tab:multiomics_{metric_label}}}\n\\end{{table}}"

    out_file = os.path.join(output_dir, f"{basename}_multiomics_{metric_label}_table.tex")
    with open(out_file, "w") as f:
        f.write(latex_str)

    print(f"Saved LaTeX table: {out_file}")


def generate_multiomics_latex_table(df, output_dir, basename):
    """
    Generate LaTeX tables for Multiomics benchmarks (AUROC and Accuracy).
    """
    output_dir = os.path.join(output_dir, "reduced_plots")

    # Determine available metrics
    auroc_metric = None
    if "roc_auc" in df.columns:
        auroc_metric = "roc_auc"
    elif "roc_auc_score" in df.columns:
        auroc_metric = "roc_auc_score"

    accuracy_metric = "accuracy" if "accuracy" in df.columns else None

    # Generate AUROC table
    if auroc_metric:
        _generate_multiomics_latex_table(df.copy(), output_dir, basename, auroc_metric, "AUROC")

    # Generate Accuracy table
    if accuracy_metric:
        _generate_multiomics_latex_table(
            df.copy(), output_dir, basename, accuracy_metric, "Accuracy"
        )


def plot_reduced_multiomics_overview_relative(df, output_dir, basename, baseline_name="v2"):
    """Reduced plot for Multiomics Overview (Relative)."""
    output_dir = os.path.join(output_dir, "reduced_plots")
    metrics = [c for c in ["roc_auc", "roc_auc_score"] if c in df.columns]
    df = clean_checkpoint_names(df)
    df = filter_to_reduced_models(df)

    if df.empty:
        return

    # Filter to max 17500 features
    if "n_features" in df.columns:
        df = df[df["n_features"] <= 17500]
        # Filter out the dataset coverage problem
        mask = (df["n_features"] > 13000) & (df["n_features"] < 14800)
        if mask.any():
            df = df[~mask]

    for metric in metrics:
        # Group by n_features and checkpoint AND dataset_name to allow paired comparison
        if "dataset_name" not in df.columns:
            # Fallback if dataset_name is missing
            df_agg = df.groupby(["n_features", "checkpoint"])[metric].mean().reset_index()
            # We can't do paired relative per dataset if we don't have dataset names,
            # so we fall back to difference of means (no error bars possible on the difference itself easily)
            # But the user asked for error bars. We'll proceed with normal aggregation but note this limitation.
            # Ideally we assume dataset_name is present for multiomics.
        else:
            df_agg = (
                df.groupby(["dataset_name", "n_features", "checkpoint"])[metric]
                .mean()
                .reset_index()
            )

        baseline_label = MODEL_CONFIG.get(baseline_name, {}).get("label", baseline_name)
        unique_checkpoints = df_agg["checkpoint"].unique()
        actual_baseline = baseline_label if baseline_label in unique_checkpoints else baseline_name

        if actual_baseline not in unique_checkpoints:
            continue

        if "dataset_name" in df_agg.columns:
            # Paired relative score calculation
            df_baseline = df_agg[df_agg["checkpoint"] == actual_baseline][
                ["dataset_name", "n_features", metric]
            ]
            df_baseline = df_baseline.rename(columns={metric: "baseline_score"})

            df_merged = pd.merge(df_agg, df_baseline, on=["dataset_name", "n_features"], how="left")
        else:
            # Fallback to averaged baseline
            df_baseline = df_agg[df_agg["checkpoint"] == actual_baseline][["n_features", metric]]
            df_baseline = df_baseline.rename(columns={metric: "baseline_score"})
            df_merged = pd.merge(df_agg, df_baseline, on="n_features", how="left")

        df_merged["relative_score"] = df_merged[metric] - df_merged["baseline_score"]

        df_plot = df_merged[df_merged["checkpoint"] != actual_baseline].copy()

        if df_plot.empty:
            continue

        palette, hue_order, dashes = get_model_style(df_plot, "checkpoint")

        with sns.plotting_context("paper", font_scale=1.6):
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.lineplot(
                data=df_plot,
                x="n_features",
                y="relative_score",
                hue="checkpoint",
                hue_order=hue_order,
                style="checkpoint",
                style_order=hue_order,
                markers=True,
                dashes=dashes if dashes else True,
                palette=palette,
                linewidth=2.5,
                markersize=8,
                errorbar="sd",
                err_kws={"alpha": 0.1},
                ax=ax,
            )
            ax.set_ylim(-0.05, 0.4)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Number of Features", fontsize=14)
            ax.set_ylabel(f"Relative {format_metric(metric)} (vs {actual_baseline})", fontsize=14)
            ax.legend(loc="best", fontsize=11, frameon=False)
            plt.tight_layout()
            save_plots(fig, output_dir, f"{basename}_{metric}_overview_relative_reduced")


def clean_basename(basename):
    for suffix in ["_benchmark_results", "_results", "_benchmark"]:
        basename = basename.replace(suffix, "")
    return basename


def aggregate_files(filename, file_list):
    """Aggregate multiple CSV files into one DataFrame."""
    dfs = []
    for f in file_list:
        try:
            temp_df = pd.read_csv(f)
        except Exception:
            continue

        if temp_df.empty:
            continue

        parent_folder = os.path.dirname(f)
        folder_name = os.path.basename(parent_folder)

        # Heuristic to find checkpoint name from folder structure
        if folder_name.startswith("sparsity"):
            parent_folder = os.path.dirname(parent_folder)
            folder_name = os.path.basename(parent_folder)
        if folder_name == "openml_widening":
            folder_name = os.path.basename(os.path.dirname(parent_folder))

        temp_df["checkpoint"] = folder_name
        dfs.append(temp_df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--compare_mode", action="store_true")
    parser.add_argument("--reduced", action="store_true", help="Generate only reduced plots")
    args = parser.parse_args()

    cwd = os.getcwd()
    base_results_dir = args.input_dir if args.input_dir else os.path.join(cwd, "analysis_results")

    if not os.path.exists(base_results_dir):
        print(f"Directory '{base_results_dir}' not found. Skipping plotting.")
        return

    if args.compare_mode:
        print(f"Running in COMPARISON MODE. Scanning {base_results_dir} recursively...")
        all_csvs = glob.glob(os.path.join(base_results_dir, "**/*.csv"), recursive=True)

        grouped_csvs = {}
        for csv_file in all_csvs:
            filename = os.path.basename(csv_file)
            grouped_csvs.setdefault(filename, []).append(csv_file)

        widening_filenames = [f for f in grouped_csvs if f.replace(".csv", "").isdigit()]
        other_filenames = [f for f in grouped_csvs if f not in widening_filenames]

        output_dir = os.path.join(base_results_dir, "comparison_plots")

        # Process widening
        widening_dfs = []
        for filename in widening_filenames:
            combined_df = aggregate_files(filename, grouped_csvs[filename])
            if combined_df is not None:
                widening_dfs.append(combined_df)

        if widening_dfs:
            all_widening_df = pd.concat(widening_dfs, ignore_index=True)
            if not args.reduced:
                plot_widening_relative(
                    all_widening_df, output_dir, "openml_widening_average", baseline_name="v2"
                )
            # Reduced plot for widening
            plot_reduced_widening_relative(
                all_widening_df.copy(), output_dir, "openml_widening_average", baseline_name="v2"
            )

        # Process others
        for filename in other_filenames:
            combined_df = aggregate_files(filename, grouped_csvs[filename])
            if combined_df is None:
                continue

            basename = clean_basename(os.path.splitext(filename)[0])

            if "multiomics" in basename.lower():
                if not args.reduced:
                    plot_multiomics(combined_df, output_dir, basename)
                    plot_multiomics_overview(combined_df, output_dir, basename)

                # Reduced plots for multiomics
                plot_reduced_multiomics_overview(combined_df.copy(), output_dir, basename)
                plot_reduced_multiomics_overview_boxplot(combined_df.copy(), output_dir, basename)
                plot_reduced_multiomics_overview_boxplot_with_baselines(
                    combined_df.copy(), output_dir, basename
                )
                plot_reduced_multiomics_overview_relative(combined_df.copy(), output_dir, basename)
                generate_multiomics_latex_table(combined_df.copy(), output_dir, basename)
            elif "grouping" in basename.lower():
                if not args.reduced:
                    plot_grouping(combined_df, output_dir, basename)
            elif "hdlss" in basename.lower():
                if not args.reduced:
                    plot_hdlss(combined_df, output_dir, basename)
                plot_reduced_hdlss_boxplot(combined_df.copy(), output_dir, basename)
            elif "openml" in basename.lower() and "widening" not in basename.lower():
                if not args.reduced:
                    plot_openml(combined_df, output_dir, basename)
                    plot_forgetting(combined_df, output_dir, basename)
            elif "snp" in basename.lower():
                if not args.reduced:
                    plot_snp(combined_df, output_dir, basename)
                    plot_snp_relative(combined_df.copy(), output_dir, basename)
                # Reduced plot for SNP
                plot_reduced_snp(combined_df.copy(), output_dir, basename)
                plot_reduced_snp_relative(combined_df.copy(), output_dir, basename)
                plot_reduced_snp_relative_polygenicity_comparison(
                    combined_df.copy(), output_dir, basename
                )
                plot_reduced_snp_average_poly(combined_df.copy(), output_dir, basename)

    else:
        # Single directory mode
        csv_files = glob.glob(os.path.join(base_results_dir, "*.csv"))
        widening_files = [
            f
            for f in glob.glob(os.path.join(base_results_dir, "**/*.csv"), recursive=True)
            if f not in csv_files
        ]
        all_files = csv_files + widening_files

        print(f"Found {len(all_files)} CSV files...")

        for csv_file in all_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            if df.empty:
                continue

            basename = clean_basename(os.path.splitext(os.path.basename(csv_file))[0])
            output_dir = os.path.dirname(csv_file)

            print(f"-- Processing {basename} --")

            if "multiomics" in basename.lower():
                plot_multiomics(df, output_dir, basename)
            elif "grouping" in basename.lower():
                plot_grouping(df, output_dir, basename)
            elif "hdlss" in basename.lower():
                plot_hdlss(df, output_dir, basename)
            elif "openml" in basename.lower() and "widening" not in basename.lower():
                plot_openml(df, output_dir, basename)
                plot_forgetting(df, output_dir, basename)
            elif "snp" in basename.lower():
                plot_snp(df, output_dir, basename)
            elif basename.isdigit() or "widening" in os.path.dirname(csv_file).lower():
                plot_widening(df, os.path.dirname(csv_file), basename, comparison_mode=False)


if __name__ == "__main__":
    main()
