# TabPFN-Wide

[![Python Versions](https://img.shields.io/pypi/pyversions/tabpfnwide.svg)](https://pypi.org/project/tabpfnwide/)
[![License](https://img.shields.io/badge/License-PriorLabs-blue.svg)](LICENSE)

> [!NOTE]
> DOI: XXX
> Authors: Christopher Kolberg, Jules Kreuer, Jonas Huurdeman, Sofiane Ouaari, Katharina Eggensperger, Nico Pfeifer


**TabPFN-Wide** is an extension of the TabPFN-2 foundation model, specifically designed for **wide datasets** (many features, few samples), such as **multi-omics** data. It allows for training and evaluating large-scale tabular models that can handle thousands of features.

This repository provides a release (v0.1.0) of the `tabpfnwide` package along with a suite of **scripts** for training, feature-smearing analysis, and biological interpretation used in the TabPFN-Wide paper.

Latter releases will include bug fixes and new features.

## Publication

The TabPFN-Wide preprint is available at [arXiv](https://arxiv.org/abs/2510.06162).

## License

The model weights and code of the tabpfnwide project are licensed under the **Prior Labs License Version 1.1**.

> [!IMPORTANT]
> The license includes an attribution requirement. If you use this work to improve an AI model, you must include "TabPFN" in the model name and display "Built with PriorLabs-TabPFN". See [LICENSE](LICENSE) for details.


## Quick Start

### Installation

**Using pip:**

```bash
pip install tabpfnwide
```

**From Source:**

```bash
pip install "tabpfnwide @ git+https://github.com/not-a-feature/TabPFN-Wide.git"
```

### Model Weights

Model weights are automatically downloaded from GitHub Releases upon first use and cached in `~/.tabpfnwide/models/`.
If you are running in an offline environment, you can manually download the `.pt` files from the [Releases page](https://github.com/not-a-feature/TabPFN-Wide/releases) and place them in that directory.

### Basic Usage

TabPFN-Wide works just like a scikit-learn classifier.

```python
from tabpfnwide.classifier import TabPFNWideClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load a 'wide' dataset (or any tabular data)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier with a wide model (e.g., handles up to 5k features)
clf = TabPFNWideClassifier(model_name="wide-v2-5k", device="cpu")

# Fit and predict
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

See `demo_run_prediction.py` for more details

### Interpreting the Model

TabPFN-Wide provides built-in tools to extract attention-based feature importance.

```python
import numpy as np

# Note: Attention maps require n_estimators=1 and features_per_group=1
clf = TabPFNWideClassifier(
    model_name="wide-v2-5k", 
    save_attention_maps=True,
    n_estimators=1,
    features_per_group=1
)
clf.fit(X_train, y_train)

# 1. Get raw feature-to-feature attention maps (list of maps per layer)
attn_per_layer = clf.get_attention_maps()
avg_attn_matrix = np.mean(attn_per_layer, axis=0)

# 2. Get direct feature importance based on label-to-feature attention
# This represents how much the model's prediction 'attends' to each input feature
importances = clf.get_attention_to_label()
```

See `demo_attention_maps.py` for more details.



## Continued Pretraining

> [!NOTE]
> Training scripts require the `dev` dependencies.

The training logic is contained in the `training/` directory. You can run training jobs using the provided python script or shell wrapper.

**Using the Python script:**

```bash
python training/train.py \
    --prior_type mlp_scm \
    --prior_max_features 100 \
    --batch_size 8
```

**Using the shell script:**

```bash
bash training/train.sh
```

### Evaluation & Analysis

```bash
bash analysis/run_analysis.sh "$CHECKPOINT_PATH" "$OUTPUT_DIR"
```

See `analysis/analysis.sbatch` for more details.

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

**Contact:**
For issues, please open a ticket on the [Issue Tracker](https://github.com/not-a-feature/TabPFN-Wide/issues).
