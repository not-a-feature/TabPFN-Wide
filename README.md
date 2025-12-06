
# TabPFN-Wide

[![Python Versions](https://img.shields.io/pypi/pyversions/tabpfnwide.svg)](https://pypi.org/project/tabpfnwide/)
[![License](https://img.shields.io/badge/License-PriorLabs-blue.svg)](LICENSE)

**TabPFN-Wide** is an extension of the TabPFN-2.5 foundation model, specifically designed for **wide datasets** (many features, few samples), such as **multi-omics** data. It allows for training and evaluating large-scale tabular models that can handle thousands of features.

This repository provides the `tabpfnwide` package along with a suite of **experimental scripts** for training, feature-smearing analysis, and biological interpretation.

---

### Quick Start

> [!TIP]
> Check out the **[Interactive Notebook](tabpfn_wide_example.ipynb)** to dive right in! It demonstrates how to initialize the model, load data, and run predictions.

---

## Installation

### Installation from Source

To install the latest version directly from GitHub:

```bash
pip install "tabpfnwide @ git+https://github.com/pfeiferAI/TabPFN-Wide.git"
```

### Local Development Installation

We recommend using `uv` for a fast and reliable development setup, but standard `pip` works as well.

**Using uv (Recommended):**

```bash
# 1. Clone the repository
git clone https://github.com/pfeiferAI/TabPFN-Wide.git
cd TabPFN-Wide

# 2. Sync dependencies
uv sync
```

**Using pip:**

```bash
# 1. Clone the repository
git clone https://github.com/pfeiferAI/TabPFN-Wide.git
cd TabPFN-Wide

# 2. Install in editable mode
# 2. Install in editable mode
pip install -e .

# Optional: Install with development dependencies (for training/evaluation)
pip install -e ".[dev]"
```

---

## Basic Usage

### Training a Model

> [!NOTE]
> Training scripts require the `dev` dependencies. Install with `pip install ".[dev]"`.

The training logic is contained in the `training/` directory. You can run training jobs using the provided python script or shell wrapper.

**Using the Python script:**

```bash
python training/train.py \
    --prior_type mlp_scm \
    --prior_max_features 100 \
    --batch_size 8
```

**(Optional) Using the shell script:**

```bash
bash training/train.sh
```

### Evaluation & Analysis

- **Attention Analysis**: Use `analysis/extract_multi_omics_attention.py` to extract interpretable attention weights from trained models.
- **Feature Reduction**: Use `analysis/multiomics_feature_reduction.py` for benchmarking on reduced feature sets.

### Ablation Studies

The repository includes scripts for systematic ablation studies to understand model behavior:

**Feature Grouping Benchmark:**

Compare TabPFN performance across different `features_per_group` settings to understand the impact of feature grouping on model accuracy:

```bash
python analysis/grouping_benchmark.py \
    --suite_id 334 \
    --max_features 100 \
    --max_instances 2000 \
    --grouping_values 1 2 3 \
    --output_file results/grouping_benchmark.csv \
    --device cuda:0
```

This will:
- Test the base TabPFN v2.5 model with different feature grouping configurations
- Run 3-fold cross-validation on each OpenML task in the specified suite
- Save detailed results and generate comparison plots automatically

---

## License

This project is licensed under the **Prior Labs License Version 1.1**.

> [!IMPORTANT]
> The license includes an attribution requirement. If you use this work to improve an AI model, you must include "TabPFN" in the model name and display "Built with PriorLabs-TabPFN". See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code or model in your research, please cite:

```bibtex
TODO

```

For the original TabPFN work, please cite:

```bibtex
@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 month={01},
 day={09},
 doi={10.1038/s41586-024-08328-6},
 publisher={Springer Nature},
 url={https://www.nature.com/articles/s41586-024-08328-6},
}
```

---

## Development & Support

**Repository Structure:**

- `tabpfnwide/`: Core installable package.
- `experiments/`: Training and evaluation scripts.
- `analysis/`: Benchmarking and interpretation tools.

**Contact:**
For issues, please open a ticket on the [Issue Tracker](https://github.com/pfeiferAI/TabPFN-Wide/issues).
