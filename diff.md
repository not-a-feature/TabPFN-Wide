diff --git a/.gemini/config.yaml b/.gemini/config.yaml
deleted file mode 100644
index 7cd1337..0000000
--- a/.gemini/config.yaml
+++ /dev/null
@@ -1,3 +0,0 @@
-code_review:
-  pull_request_opened:
-    summary: false
diff --git a/.github/CODEOWNERS b/.github/CODEOWNERS
deleted file mode 100644
index dfafa92..0000000
--- a/.github/CODEOWNERS
+++ /dev/null
@@ -1 +0,0 @@
-* @PriorLabs/opensource-review-team
diff --git a/.github/PULL_REQUEST_TEMPLATE.md b/.github/PULL_REQUEST_TEMPLATE.md
index ea4831b..ccb0442 100644
--- a/.github/PULL_REQUEST_TEMPLATE.md
+++ b/.github/PULL_REQUEST_TEMPLATE.md
@@ -1,9 +1,3 @@
-## Issue
-Please link the corresponding GitHub issue. If an issue does not already exist,
-please open one to describe the bug or feature request before creating a pull request.
-
-This allows us to discuss the proposal and helps avoid unnecessary work.
-
 ## Motivation and Context
 
 ---
@@ -27,4 +21,4 @@ This allows us to discuss the proposal and helps avoid unnecessary work.
 -   [ ] The code follows the project's style guidelines.
 -   [ ] I have considered the impact of these changes on the public API.
 
----
+---
\ No newline at end of file
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
deleted file mode 100644
index 6ba78d4..0000000
--- a/.github/workflows/ci.yml
+++ /dev/null
@@ -1,192 +0,0 @@
-name: Run tests
-on:
-  pull_request:
-  push:
-    branches: [main]
-  workflow_dispatch:
-
-concurrency:
-  group: pr-${{ github.ref }}
-  cancel-in-progress: true
-
-jobs:
-  check_python_linting:
-    name: Ruff Linting & Formatting
-    runs-on: ubuntu-latest
-    steps:
-      - uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
-      - uses: astral-sh/ruff-action@57714a7c8a2e59f32539362ba31877a1957dded1 # v3.5.1
-        with:
-          src: "./src ./tests"
-          version-file: pyproject.toml
-      - uses: astral-sh/ruff-action@57714a7c8a2e59f32539362ba31877a1957dded1 # v3.5.1
-        with:
-          src: "./src ./tests"
-          args: "format --check"
-          version-file: pyproject.toml
-
-  test_compatibility:
-    name: Test Package Compatibility
-    # This job will only start after linting succeeds
-    needs: check_python_linting
-    strategy:
-      fail-fast: false
-      matrix:
-        include:
-          - os: ubuntu-latest
-            python-version: "3.9"
-            dependency-set: lowest-direct
-          - os: macos-13 # macos-latest doesn't work with python 3.10
-            # https://github.com/actions/setup-python/issues/855
-            python-version: "3.9"
-            dependency-set: lowest-direct
-          - os: windows-latest
-            python-version: "3.9"
-            dependency-set: lowest-direct
-          # ubuntu-latest 3.13 maximum is not included here because it is executed
-          # as a separate workflow below. This is because we wish to gate the GPU
-          # workflow on ubuntu-latest 3.13 maximum, which requires it to be a
-          # separate workflow.
-          - os: macos-latest
-            python-version: "3.13"
-            dependency-set: highest
-          - os: windows-latest
-            python-version: "3.13"
-            dependency-set: highest
-    runs-on: ${{ matrix.os }}
-
-    steps:
-      - uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
-
-      - name: Set up Python ${{ matrix.python-version }}
-        uses: actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c # v6.0.0
-        with:
-          python-version: ${{ matrix.python-version }}
-          architecture: x64
-
-      - name: Install uv
-        uses: astral-sh/setup-uv@85856786d1ce8acfbcc2f13a5f3fbd6b938f9f41 # v7.1.2
-        with:
-          enable-cache: true
-
-      - name: Install dependencies
-        run: uv pip install --system --resolution ${{ matrix.dependency-set }} ".[ci]"
-        shell: bash
-
-      - name: 'Check for forbidden licenses'
-        shell: bash
-        run: |
-          licensecheck \
-            --requirements-paths pyproject.toml \
-            --show-only-failing \
-            -0
-
-      - name: Initialize submodules
-        run: git submodule update --init --recursive
-
-      - name: Run Tests (including MPS)
-        env:
-          HF_TOKEN: ${{ secrets.HF_TOKEN }}
-        if: ${{ matrix.dependency-set != 'lowest-direct' }}
-        run: pytest tests/
-
-      - name: Run Tests (without MPS)
-        if: ${{ matrix.dependency-set == 'lowest-direct' }}
-        env:
-          HF_TOKEN: ${{ secrets.HF_TOKEN }}
-          # The MPS tests are flakey on the CI for PyTorch < 2.3, possibly due to this
-          # https://github.com/pytorch/pytorch/issues/105839#issuecomment-1779116758,
-          # so disable the MPS device for these tests.
-          TABPFN_EXCLUDE_DEVICES: mps
-        run: pytest tests/
-
-  # -------------------------------------------------------------------
-  # Single Ubuntu-latest + Python 3.13 test (the gate for GPU)
-  # -------------------------------------------------------------------
-  test_ubuntu_latest_313:
-    name: Test Ubuntu-latest (Py 3.13)
-    needs: check_python_linting
-    runs-on: ubuntu-latest
-
-    steps:
-      - uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
-
-      - name: Set up Python 3.13
-        uses: actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c # v6.0.0
-        with:
-          python-version: "3.13"
-          architecture: x64
-
-      - name: Install uv
-        uses: astral-sh/setup-uv@85856786d1ce8acfbcc2f13a5f3fbd6b938f9f41 # v7.1.2
-        with:
-          enable-cache: true
-
-      - name: Install dependencies
-        run: uv pip install --system --resolution highest ".[ci]"
-        shell: bash
-
-      - name: 'Check for forbidden licenses'
-        shell: bash
-        run: |
-          licensecheck \
-            --requirements-paths pyproject.toml \
-            --show-only-failing \
-            -0
-
-      - name: Initialize submodules
-        run: git submodule update --init --recursive
-
-      - name: Run Tests
-        env:
-          HF_TOKEN: ${{ secrets.HF_TOKEN }}
-        run: pytest tests/
-
-  # -------------------------------------------------------------------
-  # GPU: To save compute we only want to execute this workflow once
-  # a CPU workflow has passed. Hence, this workflow depends on the
-  # ubuntu-latest 3.13 workflow above.
-  # -------------------------------------------------------------------
-  test_gpu:
-    name: Test on GPU
-    needs: test_ubuntu_latest_313
-
-    strategy:
-      fail-fast: false
-      matrix:
-        include:
-          - python-version: "3.9"
-            dependency-set: lowest-direct
-          - python-version: "3.13"
-            dependency-set: highest
-
-    runs-on: ubuntu-22.04-4core-gpu
-
-    steps:
-      - uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0
-
-      - name: Set up Python ${{ matrix.python-version }}
-        uses: actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c # v6.0.0
-        with:
-          python-version: ${{ matrix.python-version }}
-          architecture: x64
-
-      - name: Install uv
-        uses: astral-sh/setup-uv@85856786d1ce8acfbcc2f13a5f3fbd6b938f9f41 # v7.1.2
-        with:
-          enable-cache: true
-
-      - name: Install dependencies
-        run: uv pip install --system --resolution ${{ matrix.dependency-set }} ".[ci]"
-        shell: bash
-
-      - name: Initialize submodules
-        run: git submodule update --init --recursive
-
-      - name: Run GPU Test Suite
-        env:
-          HF_TOKEN: ${{ secrets.HF_TOKEN }}
-          CUDA_VISIBLE_DEVICES: "0"
-          # skip cpu based tests that were run separately
-          TABPFN_EXCLUDE_DEVICES: "cpu,cpu:0,mps"
-        run: pytest tests/
diff --git a/.github/workflows/pull_request.yml b/.github/workflows/pull_request.yml
new file mode 100644
index 0000000..25ed895
--- /dev/null
+++ b/.github/workflows/pull_request.yml
@@ -0,0 +1,85 @@
+name: In pull request
+on:
+  pull_request:
+
+jobs:
+  check_python_linting:
+    name: Ruff Linting & Formatting
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
+      - uses: astral-sh/ruff-action@0c50076f12c38c3d0115b7b519b54a91cb9cf0ad # v3.5.0
+        with:
+          src: "./src ./tests"
+          version-file: pyproject.toml
+      - uses: astral-sh/ruff-action@0c50076f12c38c3d0115b7b519b54a91cb9cf0ad # v3.5.0
+        with:
+          src: "./src ./tests"
+          args: "format --check"
+          version-file: pyproject.toml
+
+  test_compatibility:
+    name: Test Package Compatibility
+    strategy:
+      fail-fast: false
+      matrix:
+        include:
+          - os: ubuntu-latest
+            python-version: "3.9"
+            dependency-set: minimum
+          - os: macos-13 # macos-latest doesn't work with python 3.10
+            # https://github.com/actions/setup-python/issues/855
+            python-version: "3.9"
+            dependency-set: minimum
+          - os: windows-latest
+            python-version: "3.9"
+            dependency-set: minimum
+          - os: ubuntu-latest
+            python-version: "3.13"
+            dependency-set: maximum
+          - os: macos-latest
+            python-version: "3.13"
+            dependency-set: maximum
+          - os: windows-latest
+            python-version: "3.13"
+            dependency-set: maximum
+    runs-on: ${{ matrix.os }}
+
+    steps:
+      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
+
+      - name: Set up Python ${{ matrix.python-version }}
+        uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
+        with:
+          python-version: ${{ matrix.python-version }}
+          architecture: x64
+
+      - name: Install uv
+        uses: astral-sh/setup-uv@e92bafb6253dcd438e0484186d7669ea7a8ca1cc # v6.4.3
+        with:
+          enable-cache: true
+
+      - name: Generate requirements file
+        run: python scripts/generate_dependencies.py ${{ matrix.dependency-set }}
+
+      - name: Install dependencies
+        run: |
+          uv pip install --system --no-deps .
+          # onnx is required for onnx export tests
+          # we don't install all dev dependencies here for speed
+          uv pip install --system -r requirements.txt
+          uv pip install --system pytest psutil
+          # onnx is not supported on python 3.13 yet https://github.com/onnx/onnx/issues/6339
+          if [[ "${{ matrix.python-version }}" != "3.13" ]]; then
+            uv pip install --system onnx
+          fi
+        shell: bash
+
+      - name: Initialize submodules
+        run: git submodule update --init --recursive
+
+      - name: Run Tests (exclude MPS params)
+        env:
+          # MPS excluded since github CI device has very low MPS memory
+          TABPFN_EXCLUDE_DEVICES: "mps"
+        run: pytest tests/
diff --git a/.gitignore b/.gitignore
index 3067b50..5c2d72e 100644
--- a/.gitignore
+++ b/.gitignore
@@ -163,5 +163,4 @@ cython_debug/
 ./src/.DS_Store
 
 # Claude AI assistant
-CLAUDE.md
-uv.lock
+CLAUDE.md
\ No newline at end of file
diff --git a/.pre-commit-config.yaml b/.pre-commit-config.yaml
index b9549ae..b3019dd 100644
--- a/.pre-commit-config.yaml
+++ b/.pre-commit-config.yaml
@@ -50,7 +50,7 @@ repos:
     hooks:
       - id: commitizen
   - repo: https://github.com/astral-sh/ruff-pre-commit
-    rev: v0.14.0 # This version must be the same as in pyproject.toml
+    rev: v0.8.6 # This version must be the same as in pyproject.toml
     hooks:
       - id: ruff
         args: [--fix, --exit-non-zero-on-fix, --no-cache]
diff --git a/CHANGELOG.md b/CHANGELOG.md
index 4b5fcc5..5e741eb 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -5,94 +5,21 @@ All notable changes to this project will be documented in this file.
 The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
 and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
 
-## [6.0.6] - 2025-11-10
+## [Unreleased]
 
 ### Added
-- Add a link to the gated model docs to the error message [#613](https://github.com/PriorLabs/TabPFN/pull/613)
-- Anonymously report on used `model_path` and `model_version` [#611](https://github.com/PriorLabs/TabPFN/pull/611)
-
-## [6.0.1] - 2025-11-06
-
-### Changed
-
-- Updated automatic selection of memory saving mode to improve fit + predict speed [#605](https://github.com/PriorLabs/TabPFN/pull/605)
-
-## [6.0.0] - 2025-11-06
-
-### Added
-
-- Released TabPFN-2.5, a strong improvement over TabPFNv2 scaling to datasets with up to 50,000 samples and 2,000 features (more details [here](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)). This is used by default when using package version 6.0.0 and higher. To use the previous version, use `from tabpfn.constants import ModelVersion; TabPFNClassifier.create_default_for_version(ModelVersion.V2)`. Note that TabPFN-2.5 is released under a new [TABPFN-2.5 Non-Commercial License v1.0 license](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE).
-
-### Changed
-
-- Deprecated the parameters `TabPFNClassifier(n_jobs=...)` and
-  `TabPFNRegressor(n_jobs=...)` which had no effect, and replaced them with
-  functioning `n_preprocessing_jobs`. We strongly recommend using the default value of
-  `1`. [#555](https://github.com/PriorLabs/TabPFN/pull/555)
-- Introduced interface to use `TabPFNClassifier` and `TabPFNRegressor` with multiple models in an ensemble. [#557](https://github.com/PriorLabs/TabPFN/pull/557)
-- Fix precision of model outputs in the case when `softmax_temperature=1.0` [#569](https://github.com/PriorLabs/TabPFN/pull/569)
-- Rename `tabpfn.config.ModelInterfaceConfig` to `tabpfn.inference_config.InferenceConfig` [#575](https://github.com/PriorLabs/TabPFN/pull/575)
-- Add option to `TabPFNClassifier` to calibrate probabilities and tune decision thresholds for a specified metric. The feature can be used by specifying `eval_metric` and `tuning_config` during initialization [#218](https://github.com/PriorLabs/TabPFN-private/pull/218)
-- Change `ensure_y_numeric=False` for `TabPFNRegressor` to `True` - need to validate `y_train` contains numerics.
-
-## [2.2.1] - 2025-09-17
 
 ### Changed
 
-- Fixed bug on multi-GPU systems leading to worse results
-
-## [2.2.0] - 2025-09-15
-
-### Added
-
-### Changed
-
-- Refactored preprocessing-related code [#503](https://github.com/PriorLabs/TabPFN/pull/503).
-- Improved speed of `QuantileTransformer` for sample sizes larger 10k. This change also leads to subtle changes (improving the outcomes of the transformer slightly) at large sample sizes. [#503](https://github.com/PriorLabs/TabPFN/pull/503).
-- @safaricd Clarified details of anonymous usage telemetry collection.
-
 ### Bug Fixes
 
-## [2.1.4] - 2025-09-11 - **yanked**
-
-### Added
-
-### Changed
-
-- @benraha Improved the inference speed on CPU significantly [#459](https://github.com/PriorLabs/TabPFN/pull/459).
-- @benraha Added a fast-path for the column selection in RemoveEmptyFeaturesEncoderStep [#468](https://github.com/PriorLabs/TabPFN/pull/468).
-- @safaricd Added anonymous usage analytics [#499](https://github.com/PriorLabs/TabPFN/pull/499)
-- `TabPFNClassifier/Regressor.device_` has been replaced with `.devices_` [#496](https://github.com/PriorLabs/TabPFN/pull/496).
-
-### Bug Fixes
-
-## [2.1.3] - 2025-08-26
-
-### Added
-
-- Added several new finetuned model checkpoints. ([#462](https://github.com/PriorLabs/TabPFN/pull/462))
-
-### Changed
-
-### Bug Fixes
-
-- Current infer categoricals crashes in case user tries to pass a feature as input that contains str and nan values. ([#432](https://github.com/PriorLabs/TabPFN/pull/432))
-- Fixed a validation error that occurred when a `.env` file contained settings from other applications. ([#446](https://github.com/PriorLabs/TabPFN/pull/446))
-- Fixed a crash on PyTorch versions older than 2.5 by correctly detecting Grouped-Query Attention (GQA) support. ([#438](https://github.com/PriorLabs/TabPFN/pull/438))
-
-## [2.1.2] - 2025-08-03
-
-- No changes -
-
 ## [2.1.1] - 2025-08-03
 
 ### Added
-
 - Added a new `predict_logits()` method to `TabPFNClassifier` to return raw model outputs (logits). This is useful for model explainability tasks (e.g., with SHAP) that benefit from unnormalized, additive outputs.
 - Support for MPS device: TabPFN can run on local Apple MPS Accelerator.
 
 ### Changed
-
 - Increased the default value of the `n_estimators` parameter in `TabPFNClassifier` from `4` to `8`. This change aims to improve average accuracy by default, with the trade-off of increased inference time and memory usage. ([#384](https://github.com/PriorLabs/TabPFN/pull/384))
 - Refactored the internal prediction logic for `TabPFNClassifier` for improved clarity, modularity, and maintainability.
 - Regression finetuning outputs are renamed to more clearly reflect their purpose.
@@ -100,7 +27,6 @@ and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0
 - Classifier finetunging now operates on the logits directly.
 
 ### Bug fix
-
 - @benraha fixed a bug with differentiable inputs to the TabPFNClassifer.
 - @zhengaq fixed a bug when a row was completely consisting of missing values.
 - @rosenyu304 fixed a bug with the random number generator for old sklearn versions.
@@ -108,11 +34,9 @@ and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0
 ## [2.1.0] - 2025-07-04
 
 ### Changed
-
 - **New Default Model**: The default classifier model has been updated to a new finetuned version (`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`) to improve out-of-the-box performance.
 - **Overhauled Examples**: The finetuning examples (`finetune_classifier.py`, `finetune_regressor.py`) have been completely rewritten with a clearer structure, centralized configuration, and more robust evaluation.
 - Simplified `ignore_pretraining_limits` behavior by removing redundant warnings when the flag is enabled.
 
 ### Fixed
-
 - The model now automatically switches between `fit_mode='batched'` and standard modes when calling `fit()` and `fit_from_preprocessed()`. This prevents crashes and provides a smoother finetuning experience by logging a warning instead of raising an error.
diff --git a/README.md b/README.md
index 8b2b3cc..c34f1d1 100644
--- a/README.md
+++ b/README.md
@@ -9,7 +9,7 @@
 
 <img src="https://github.com/PriorLabs/tabpfn-extensions/blob/main/tabpfn_summary.webp" width="80%" alt="TabPFN Summary">
 
-## Quick Start
+## ≡ƒÅü Quick Start
 
 ### Interactive Notebook Tutorial
 > [!TIP]
@@ -48,19 +48,15 @@ from sklearn.metrics import accuracy_score, roc_auc_score
 from sklearn.model_selection import train_test_split
 
 from tabpfn import TabPFNClassifier
-from tabpfn.constants import ModelVersion
 
 # Load data
 X, y = load_breast_cancer(return_X_y=True)
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
 
 # Initialize a classifier
-clf = TabPFNClassifier()  # Uses TabPFN 2.5 weights, finetuned on real data.
-# To use TabPFN v2:
-# clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
+clf = TabPFNClassifier()
 clf.fit(X_train, y_train)
 
-
 # Predict probabilities
 prediction_probabilities = clf.predict_proba(X_test)
 print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
@@ -77,7 +73,6 @@ from sklearn.metrics import mean_squared_error, r2_score
 from sklearn.model_selection import train_test_split
 
 from tabpfn import TabPFNRegressor
-from tabpfn.constants import ModelVersion
 
 # Load Boston Housing data
 df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
@@ -88,9 +83,7 @@ y = df.target.astype(float)  # Ensure target is float for regression
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
 
 # Initialize the regressor
-regressor = TabPFNRegressor()  # Uses TabPFN-2.5 weights, trained on synthetic data only.
-# To use TabPFN v2:
-# regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
+regressor = TabPFNRegressor()
 regressor.fit(X_train, y_train)
 
 # Predict on the test set
@@ -104,7 +97,27 @@ print("Mean Squared Error (MSE):", mse)
 print("R┬▓ Score:", r2)
 ```
 
-## TabPFN Ecosystem
+### Best Results
+
+For optimal performance, use the `AutoTabPFNClassifier` or `AutoTabPFNRegressor` for post-hoc ensembling. These can be found in the [TabPFN Extensions](https://github.com/PriorLabs/tabpfn-extensions) repository. Post-hoc ensembling combines multiple TabPFN models into an ensemble.
+
+**Steps for Best Results:**
+1. Install the extensions:
+   ```bash
+   git clone https://github.com/priorlabs/tabpfn-extensions.git
+   pip install -e tabpfn-extensions
+   ```
+
+2.
+   ```python
+   from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
+
+   clf = AutoTabPFNClassifier(max_time=120, device="cuda") # 120 seconds tuning time
+   clf.fit(X_train, y_train)
+   predictions = clf.predict(X_test)
+   ```
+
+## ≡ƒîÉ TabPFN Ecosystem
 
 Choose the right TabPFN implementation for your needs:
 
@@ -114,15 +127,15 @@ Choose the right TabPFN implementation for your needs:
 - **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**
   A powerful companion repository packed with advanced utilities, integrations, and features - great place to contribute:
 
-  -  **`interpretability`**: Gain insights with SHAP-based explanations, feature importance, and selection tools.
-  -  **`unsupervised`**: Tools for outlier detection and synthetic tabular data generation.
-  -  **`embeddings`**: Extract and use TabPFNΓÇÖs internal learned embeddings for downstream tasks or analysis.
-  -  **`many_class`**: Handle multi-class classification problems that exceed TabPFN's built-in class limit.
-  -  **`rf_pfn`**: Combine TabPFN with traditional models like Random Forests for hybrid approaches.
-  -  **`hpo`**: Automated hyperparameter optimization tailored to TabPFN.
-  -  **`post_hoc_ensembles`**: Boost performance by ensembling multiple TabPFN models post-training.
+  - ≡ƒöì **`interpretability`**: Gain insights with SHAP-based explanations, feature importance, and selection tools.
+  - ≡ƒò╡∩╕ÅΓÇìΓÖé∩╕Å **`unsupervised`**: Tools for outlier detection and synthetic tabular data generation.
+  - ≡ƒº¼ **`embeddings`**: Extract and use TabPFNΓÇÖs internal learned embeddings for downstream tasks or analysis.
+  - ≡ƒºá **`many_class`**: Handle multi-class classification problems that exceed TabPFN's built-in class limit.
+  - ≡ƒî▓ **`rf_pfn`**: Combine TabPFN with traditional models like Random Forests for hybrid approaches.
+  - ΓÜÖ∩╕Å **`hpo`**: Automated hyperparameter optimization tailored to TabPFN.
+  - ≡ƒöü **`post_hoc_ensembles`**: Boost performance by ensembling multiple TabPFN models post-training.
 
-  To install:
+  Γ£¿ To install:
   ```bash
   git clone https://github.com/priorlabs/tabpfn-extensions.git
   pip install -e tabpfn-extensions
@@ -134,162 +147,11 @@ Choose the right TabPFN implementation for your needs:
 - **[TabPFN UX](https://ux.priorlabs.ai)**
   No-code graphical interface to explore TabPFN capabilitiesΓÇöideal for business users and prototyping.
 
-## TabPFN Workflow at a Glance
-Follow this decision tree to build your model and choose the right extensions from our ecosystem. It walks you through critical questions about your data, hardware, and performance needs, guiding you to the best solution for your specific use case.
-
-```mermaid
----
-config:
-  theme: 'default'
-  themeVariables:
-    edgeLabelBackground: 'white'
----
-graph LR
-    %% 1. DEFINE COLOR SCHEME & STYLES
-    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#333;
-    classDef start_node fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#333;
-    classDef process_node fill:#e0f2f1,stroke:#00796b,stroke-width:2px,color:#333;
-    classDef decision_node fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#333;
-
-    style Infrastructure fill:#fff,stroke:#ccc,stroke-width:5px;
-    style Unsupervised fill:#fff,stroke:#ccc,stroke-width:5px;
-    style Data fill:#fff,stroke:#ccc,stroke-width:5px;
-    style Performance fill:#fff,stroke:#ccc,stroke-width:5px;
-    style Interpretability fill:#fff,stroke:#ccc,stroke-width:5px;
-
-    %% 2. DEFINE GRAPH STRUCTURE
-    subgraph Infrastructure
-        start((Start)) --> gpu_check["GPU available?"];
-        gpu_check -- Yes --> local_version["Use TabPFN<br/>(local PyTorch)"];
-        gpu_check -- No --> api_client["Use TabPFN-Client<br/>(cloud API)"];
-        task_type["What is<br/>your task?"]
-    end
-
-    local_version --> task_type
-    api_client --> task_type
-
-    end_node((Workflow<br/>Complete));
-
-    subgraph Unsupervised
-        unsupervised_type["Select<br/>Unsupervised Task"];
-        unsupervised_type --> imputation["Imputation"]
-        unsupervised_type --> data_gen["Data<br/>Generation"];
-        unsupervised_type --> tabebm["Data<br/>Augmentation"];
-        unsupervised_type --> density["Outlier<br/>Detection"];
-        unsupervised_type --> embedding["Get<br/>Embeddings"];
-    end
-
-
-    subgraph Data
-        data_check["Data Checks"];
-        model_choice["Samples > 10k or<br/>Classes > 10?"]
-        data_check -- "Table Contains Text Data?" --> api_backend_note["Note: API client has<br/>native text support"];
-        api_backend_note --> model_choice;
-        data_check -- "Time-Series Data?" --> ts_features["Use Time-Series<br/>Features"];
-        ts_features --> model_choice;
-        data_check -- "Purely Tabular" --> model_choice;
-        model_choice -- "No" --> finetune_check;
-        model_choice -- "Yes, >10k samples" --> subsample["Large Datasets Guide<br/>"];
-        model_choice -- "Yes, >10 classes" --> many_class["Many-Class<br/>Method"];
-    end
-
-    subgraph Performance
-        finetune_check["Need<br/>Finetuning?"];
-        performance_check["Need Even Better Performance?"];
-        speed_check["Need faster inference<br/>at prediction time?"];
-        kv_cache["Enable KV Cache<br/>(fit_mode='fit_with_cache')<br/><small>Faster predict; +Memory ~O(N├ùF)</small>"];
-        tuning_complete["Tuning Complete"];
-
-        finetune_check -- Yes --> finetuning["Finetuning"];
-        finetune_check -- No --> performance_check;
-
-        finetuning --> performance_check;
-
-        performance_check -- No --> tuning_complete;
-        performance_check -- Yes --> hpo["HPO"];
-        performance_check -- Yes --> post_hoc["Post-Hoc<br/>Ensembling"];
-        performance_check -- Yes --> more_estimators["More<br/>Estimators"];
-        performance_check -- Yes --> speed_check;
-
-        speed_check -- Yes --> kv_cache;
-        speed_check -- No --> tuning_complete;
-
-        hpo --> tuning_complete;
-        post_hoc --> tuning_complete;
-        more_estimators --> tuning_complete;
-        kv_cache --> tuning_complete;
-    end
-
-    subgraph Interpretability
-
-        tuning_complete --> interpretability_check;
-
-        interpretability_check["Need<br/>Interpretability?"];
-
-        interpretability_check --> feature_selection["Feature Selection"];
-        interpretability_check --> partial_dependence["Partial Dependence Plots"];
-        interpretability_check --> shapley["Explain with<br/>SHAP"];
-        interpretability_check --> shap_iq["Explain with<br/>SHAP IQ"];
-        interpretability_check -- No --> end_node;
-
-        feature_selection --> end_node;
-        partial_dependence --> end_node;
-        shapley --> end_node;
-        shap_iq --> end_node;
-
-    end
-
-    %% 3. LINK SUBGRAPHS AND PATHS
-    task_type -- "Classification or Regression" --> data_check;
-    task_type -- "Unsupervised" --> unsupervised_type;
-
-    subsample --> finetune_check;
-    many_class --> finetune_check;
-
-    %% 4. APPLY STYLES
-    class start,end_node start_node;
-    class local_version,api_client,imputation,data_gen,tabebm,density,embedding,api_backend_note,ts_features,subsample,many_class,finetuning,feature_selection,partial_dependence,shapley,shap_iq,hpo,post_hoc,more_estimators,kv_cache process_node;
-    class gpu_check,task_type,unsupervised_type,data_check,model_choice,finetune_check,interpretability_check,performance_check,speed_check decision_node;
-    class tuning_complete process_node;
-
-    %% 5. ADD CLICKABLE LINKS (INCLUDING KV CACHE EXAMPLE)
-    click local_version "https://github.com/PriorLabs/TabPFN" "TabPFN Backend Options" _blank
-    click api_client "https://github.com/PriorLabs/tabpfn-client" "TabPFN API Client" _blank
-    click api_backend_note "https://github.com/PriorLabs/tabpfn-client" "TabPFN API Backend" _blank
-    click unsupervised_type "https://github.com/PriorLabs/tabpfn-extensions" "TabPFN Extensions" _blank
-    click imputation "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/imputation.py" "TabPFN Imputation Example" _blank
-    click data_gen "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/generate_data.py" "TabPFN Data Generation Example" _blank
-    click tabebm "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/tabebm/tabebm_augment_real_world_data.ipynb" "TabEBM Data Augmentation Example" _blank
-    click density "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/density_estimation_outlier_detection.py" "TabPFN Density Estimation/Outlier Detection Example" _blank
-    click embedding "https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/embedding" "TabPFN Embedding Example" _blank
-    click ts_features "https://github.com/PriorLabs/tabpfn-time-series" "TabPFN Time-Series Example" _blank
-    click many_class "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/many_class/many_class_classifier_example.py" "Many Class Example" _blank
-    click finetuning "https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_classifier.py" "Finetuning Example" _blank
-    click feature_selection "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/feature_selection.py" "Feature Selection Example" _blank
-    click partial_dependence "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/pdp_example.py" "Partial Dependence Plots Example" _blank
-    click shapley "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shap_example.py" "Shapley Values Example" _blank
-    click shap_iq "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shapiq_example.py" "SHAP IQ Example" _blank
-    click post_hoc "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/phe/phe_example.py" "Post-Hoc Ensemble Example" _blank
-    click hpo "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/hpo/tuned_tabpfn.py" "HPO Example" _blank
-    click subsample "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/large_datasets/large_datasets_example.py" "Large Datasets Example" _blank
-    click kv_cache "https://github.com/PriorLabs/TabPFN/blob/main/examples/kv_cache_fast_prediction.py" "KV Cache Fast Prediction Example" _blank
-
-```
-
-## License
-
-The TabPFN-2.5 model weights are licensed under a [non-commercial license](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE). These are used by default.
-
-The code and TabPFN-2 model weights are licensed under Prior Labs License (Apache 2.0 with additional attribution requirement): [here](https://priorlabs.ai/tabpfn-license/). To use the v2 model weights, instantiate your model as follows:
-
-```
-from tabpfn.constants import ModelVersion
-tabpfn_v2 = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
-```
+## ≡ƒô£ License
 
+Prior Labs License (Apache 2.0 with additional attribution requirement): [here](https://priorlabs.ai/tabpfn-license/)
 
-
-## Join Our Community
+## ≡ƒñ¥ Join Our Community
 
 We're building the future of tabular machine learning and would love your involvement:
 
@@ -300,12 +162,12 @@ We're building the future of tabular machine learning and would love your involv
 
 2. **Contribute**:
    - Report bugs or request features
-   - Submit pull requests (please make sure to open an issue discussing the feature/bug first if none exists)
+   - Submit pull requests
    - Share your research and use cases
 
 3. **Stay Updated**: Star the repo and join Discord for the latest updates
 
-## Citation
+## ≡ƒôÜ Citation
 
 You can read our paper explaining TabPFN [here](https://doi.org/10.1038/s41586-024-08328-6).
 
@@ -339,19 +201,13 @@ You can read our paper explaining TabPFN [here](https://doi.org/10.1038/s41586-0
 ### **Usage & Compatibility**
 
 **Q: What dataset sizes work best with TabPFN?**
-A: TabPFN-2.5 is optimized for **datasets up to 50,000 rows**. For larger datasets, consider using **Random Forest preprocessing** or other extensions. See our [Colab notebook](https://colab.research.google.com/drive/154SoIzNW1LHBWyrxNwmBqtFAr1uZRZ6a#scrollTo=OwaXfEIWlhC8) for strategies.
+A: TabPFN is optimized for **datasets up to 10,000 rows**. For larger datasets, consider using **Random Forest preprocessing** or other extensions. See our [Colab notebook](https://colab.research.google.com/drive/154SoIzNW1LHBWyrxNwmBqtFAr1uZRZ6a#scrollTo=OwaXfEIWlhC8) for strategies.
 
 **Q: Why can't I use TabPFN with Python 3.8?**
-A: TabPFN requires **Python 3.9+** due to newer language features. Compatible versions: **3.9, 3.10, 3.11, 3.12, 3.13**.
+A: TabPFN v2 requires **Python 3.9+** due to newer language features. Compatible versions: **3.9, 3.10, 3.11, 3.12, 3.13**.
 
 ### **Installation & Setup**
 
-**Q: How do I get access to TabPFN-2.5?**
-
-Visit [https://huggingface.co/Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5) and accept the license terms. If access via huggingface is not an option for you, please contact us at [`sales@priorlabs.ai`](mailto:sales@priorlabs.ai).
-
-Downloading the model requires your machine to be logged into Hugging Face. To do so, run `hf auth login` in your terminal, see the [huggingface documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) for details..
-
 **Q: How do I use TabPFN without an internet connection?**
 
 TabPFN automatically downloads model weights when first used. For offline usage:
@@ -370,8 +226,8 @@ This script will download the main classifier and regressor models, as well as a
 **Manual Download**
 
 1. Download the model files manually from HuggingFace:
-   - Classifier: [tabpfn-v2.5-classifier-v2.5_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/tabpfn-v2.5-classifier-v2.5_default.ckpt) (Note: the classifier default uses the model fine-tuned on real data).
-   - Regressor: [tabpfn-v2.5-regressor-v2.5_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/tabpfn-v2.5-regressor-v2.5_default.ckpt)
+   - Classifier: [tabpfn-v2-classifier.ckpt](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier.ckpt)
+   - Regressor: [tabpfn-v2-regressor.ckpt](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt)
 
 2. Place the file in one of these locations:
    - Specify directly: `TabPFNClassifier(model_path="/path/to/model.ckpt")`
@@ -440,12 +296,11 @@ A: **Yes!**
 A: Best practices:
 - Use **AutoTabPFNClassifier** from [TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions) for post-hoc ensembling
 - Feature engineering: Add domain-specific features to improve model performance
-
 Not effective:
-- Adapt feature scaling
-- Convert categorical features to numerical values (e.g., one-hot encoding)
+  - Adapt feature scaling
+  - Convert categorical features to numerical values (e.g., one-hot encoding)
 
-## Development
+## ≡ƒ¢á∩╕Å Development
 
 1. Setup environment:
 ```bash
@@ -467,23 +322,6 @@ pre-commit run --all-files
 pytest tests/
 ```
 
-## Anonymized Telemetry
-
-This project collects fully anonymous usage telemetry with an option to opt-out of any telemetry or opt-in to extended telemetry.
-
-The data is used exclusively to help us provide stability to the relevant products and compute environments and guide future improvements.
-
-- **No personal data is collected**
-- **No code, model inputs, or outputs are ever sent**
-- **Data is strictly anonymous and cannot be linked to individuals**
-
-For details on telemetry, please see our [Telemetry Reference](https://github.com/PriorLabs/TabPFN/blob/main/TELEMETRY.md) and our [Privacy Policy](https://priorlabs.ai/privacy_policy/).
-
-**To opt out**, set the following environment variable:
-
-```bash
-export TABPFN_DISABLE_TELEMETRY=1
-```
 ---
 
 Built with Γ¥ñ∩╕Å by [Prior Labs](https://priorlabs.ai) - Copyright (c) 2025 Prior Labs GmbH
diff --git a/TELEMETRY.md b/TELEMETRY.md
deleted file mode 100644
index 0c4f113..0000000
--- a/TELEMETRY.md
+++ /dev/null
@@ -1,65 +0,0 @@
-# Telemetry
-
-This project includes lightweight, anonymous telemetry to help us improve TabPFN.  
-We've designed this with two goals in mind:
-
-1. Γ£à Be **fully GDPR-compliant** (no personal data, no sensitive data, no surprises)  
-2. Γ£à Be **OSS-friendly and transparent** about what we track and why  
-
-If you'd rather not send telemetry, you can always opt out (see **Opting out**).
-
----
-
-## What we collect
-
-We only gather **very high-level usage signals** ΓÇö enough to guide development, never enough to identify you or your data.  
-
-Here's the full list:
-
-### Events
-- `ping` ΓÇô sent when models initialize, used to check liveness  
-- `fit_called` ΓÇô sent when you call `fit`  
-- `predict_called` ΓÇô sent when you call `predict`  
-
-### Metadata (all events)
-- `python_version` ΓÇô version of Python you're running  
-- `tabpfn_version` ΓÇô TabPFN package version  
-- `timestamp` ΓÇô time of the event  
-
-### Extra metadata (`fit` / `predict` only)
-- `task` ΓÇô whether the call was for **classification** or **regression**  
-- `num_rows` ΓÇô *rounded* number of rows in your dataset  
-- `num_columns` ΓÇô *rounded* number of columns in your dataset  
-- `duration_ms` ΓÇô time it took to complete the call  
-
----
-
-## How we protect your privacy
-
-- **No inputs, no outputs, no code** ever leave your machine.  
-- **No personal data** is collected.  
-- Dataset shapes are **rounded into ranges** (e.g. `(953, 17)` ΓåÆ `(1000, 20)`) so exact dimensionalities can't be linked back to you.  
-- The data is strictly anonymous ΓÇö it cannot be tied to individuals, projects, or datasets.  
-
-This approach lets us understand dataset *patterns* (e.g. "most users run with ~1k features") while ensuring no one's data is exposed.  
-
----
-
-## Why we collect telemetry?
-
-Open-source projects don't get much feedback unless people file issues. Telemetry helps us:  
-- See which parts of TabPFN are most used (fit vs predict, classification vs regression)  
-- Detect performance bottlenecks and stability issues  
-- Prioritize improvements that benefit the most users  
-
-This information goes directly into **making TabPFN better** for the community.  
-
----
-
-## Opting out
-
-Don't want to send telemetry? No problem ΓÇö just set the environment variable:
-
-```bash
-export TABPFN_DISABLE_TELEMETRY=1
-```
diff --git a/examples/finetune_classifier.py b/examples/finetune_classifier.py
index dac112b..315c3cb 100644
--- a/examples/finetune_classifier.py
+++ b/examples/finetune_classifier.py
@@ -64,16 +64,10 @@ def setup_model_and_optimizer(config: dict) -> tuple[TabPFNClassifier, Optimizer
         **classifier_config, fit_mode="batched", differentiable_input=False
     )
     classifier._initialize_model_variables()
-
-    if len(classifier.models_) > 1:
-        raise ValueError(
-            f"Your TabPFNClassifier uses multiple models ({len(classifier.models_)}). "
-            "Finetuning is not supported for multiple models. Please use a single model."
-        )
-    model = classifier.models_[0]
-
     # Optimizer uses finetuning-specific learning rate
-    optimizer = Adam(model.parameters(), lr=config["finetuning"]["learning_rate"])
+    optimizer = Adam(
+        classifier.model_.parameters(), lr=config["finetuning"]["learning_rate"]
+    )
 
     print(f"Using device: {config['device']}")
     print(f"Optimizer: Adam, Finetuning LR: {config['finetuning']['learning_rate']}")
@@ -108,7 +102,7 @@ def evaluate_model(
     return roc_auc, log_loss_score
 
 
-def main() -> None:
+def main():
     """Main function to configure and run the finetuning workflow."""
     # --- Master Configuration ---
     config = {
diff --git a/examples/finetune_regressor.py b/examples/finetune_regressor.py
index f92e88b..6171424 100644
--- a/examples/finetune_regressor.py
+++ b/examples/finetune_regressor.py
@@ -103,7 +103,7 @@ def evaluate_regressor(
     return mse, mae, r2
 
 
-def main() -> None:
+def main():
     """Main function to configure and run the finetuning workflow."""
     # --- Master Configuration ---
     # This improved structure separates general settings from finetuning hyperparameters.
@@ -145,14 +145,6 @@ def main() -> None:
     X_train, X_test, y_train, y_test = prepare_data(config)
     regressor, regressor_config = setup_regressor(config)
 
-    if len(regressor.models_) > 1:
-        raise ValueError(
-            f"Your TabPFNRegressor uses multiple models ({len(regressor.models_)}). "
-            "Finetuning is not supported for multiple models. Please use a single model."
-        )
-
-    model = regressor.models_[0]
-
     splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
     # Note: `max_data_size` corresponds to the finetuning `batch_size` in the config
     training_datasets = regressor.get_preprocessed_datasets(
@@ -165,7 +157,9 @@ def main() -> None:
     )
 
     # Optimizer must be created AFTER get_preprocessed_datasets, which initializes the model
-    optimizer = Adam(model.parameters(), lr=config["finetuning"]["learning_rate"])
+    optimizer = Adam(
+        regressor.model_.parameters(), lr=config["finetuning"]["learning_rate"]
+    )
     print(
         f"--- Optimizer Initialized: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
     )
@@ -187,28 +181,25 @@ def main() -> None:
             for data_batch in progress_bar:
                 optimizer.zero_grad()
                 (
-                    X_trains_preprocessed,
-                    X_tests_preprocessed,
-                    y_trains_znorm,
-                    y_test_znorm,
+                    X_trains_p,
+                    X_tests_p,
+                    y_trains_p,
+                    y_test_std,
                     cat_ixs,
                     confs,
-                    raw_space_bardist_,
-                    znorm_space_bardist_,
+                    norm_bardist,
+                    bardist,
                     _,
-                    _y_test_raw,
+                    batch_y_test_raw,
                 ) = data_batch
 
-                regressor.raw_space_bardist_ = raw_space_bardist_[0]
-                regressor.bardist_ = znorm_space_bardist_[0]
-                regressor.fit_from_preprocessed(
-                    X_trains_preprocessed, y_trains_znorm, cat_ixs, confs
-                )
-                logits, _, _ = regressor.forward(X_tests_preprocessed)
+                regressor.normalized_bardist_ = norm_bardist[0]
+                regressor.fit_from_preprocessed(X_trains_p, y_trains_p, cat_ixs, confs)
+                logits, _, _ = regressor.forward(X_tests_p)
 
                 # For regression, the loss function is part of the preprocessed data
-                loss_fn = znorm_space_bardist_[0]
-                y_target = y_test_znorm
+                loss_fn = norm_bardist[0]
+                y_target = y_test_std
 
                 loss = loss_fn(logits, y_target.to(config["device"])).mean()
                 loss.backward()
diff --git a/examples/kv_cache_fast_prediction.py b/examples/kv_cache_fast_prediction.py
deleted file mode 100644
index cd3fe66..0000000
--- a/examples/kv_cache_fast_prediction.py
+++ /dev/null
@@ -1,56 +0,0 @@
-#  Copyright (c) Prior Labs GmbH 2025.
-"""TabPFN with KV cache vs. without on binary classification (synthetic).
-
-`fit_mode="fit_with_cache"` builds a key-value (KV) cache *during* `fit`.
-This front-loads the cost of computing the training-set representation so
-`predict`/`predict_proba` run fasterΓÇöespecially when:
-  ΓÇó the training set is large, and/or
-  ΓÇó the test:train ratio is small (few predictions per many training points).
-
-Trade-off: additional memory roughly O(N_samples * N_features) to hold the cache.
-Implications:
-  ΓÇó Expect *slower* `fit` but *faster* `predict`/`predict_proba`.
-  ΓÇó Benefit grows with train size and repeated inference (CV folds, batch eval, etc.).
-"""
-
-import time
-
-from sklearn.datasets import make_classification
-from sklearn.metrics import accuracy_score
-from sklearn.model_selection import train_test_split
-
-from tabpfn import TabPFNClassifier
-
-# Load data
-X, y = make_classification(n_samples=5000, n_features=20, random_state=42, n_classes=2)
-
-X_train, X_test, y_train, y_test = train_test_split(
-    X, y, test_size=0.05, random_state=42, stratify=y
-)
-
-
-def bench(clf: TabPFNClassifier, name: str) -> None:
-    t0 = time.perf_counter()
-    clf.fit(X_train, y_train)
-    t_fit = time.perf_counter() - t0
-
-    # First inference already benefits if cache was built during fit.
-    t1 = time.perf_counter()
-    preds = clf.predict(X_test)
-    t_pred = time.perf_counter() - t1
-
-    print(
-        f"[{name}] fit: {t_fit:.4f}s | predict: {t_pred:.4f}s "
-        f"| Acc: {accuracy_score(y_test, preds):.3f} "
-    )
-
-
-# Baseline: no cache
-clf_no_cache = (
-    TabPFNClassifier()
-)  # default mode (training part recomputed at predict time)
-bench(clf_no_cache, "no_cache")
-
-# With KV cache: cache is built during `fit`, so first predict is faster
-clf_kv = TabPFNClassifier(fit_mode="fit_with_cache")
-bench(clf_kv, "kv_cache")
diff --git a/examples/tabpfn_with_tuning.py b/examples/tabpfn_with_tuning.py
deleted file mode 100644
index 18a496c..0000000
--- a/examples/tabpfn_with_tuning.py
+++ /dev/null
@@ -1,56 +0,0 @@
-#  Copyright (c) Prior Labs GmbH 2025.
-"""Example of using TabPFN for binary classification with an eval_metric and tuning.
-
-This example demonstrates how to calibrate and tune the predictions
-of a TabPFNClassifier with an eval_metric and tuning_config.
-"""
-
-from sklearn.datasets import make_classification
-from sklearn.metrics import f1_score
-from sklearn.model_selection import StratifiedShuffleSplit
-
-from tabpfn import TabPFNClassifier
-
-MINORITY_FRAC = 0.04
-
-# Generate an imbalanced dataset
-X, y = make_classification(
-    n_samples=3_000,
-    n_features=4,
-    n_classes=2,
-    n_informative=4,
-    n_redundant=0,
-    weights=[float(1.0 - MINORITY_FRAC), float(MINORITY_FRAC)],
-    random_state=42,
-)
-
-print(f"Generated dataset with imbalance ratio: {len(y[y == 1]) / len(y[y == 0]):.3f}")
-
-stratified_splitter = StratifiedShuffleSplit(
-    n_splits=1,
-    test_size=0.33,
-    random_state=42,
-)
-train_index, test_index = next(stratified_splitter.split(X, y))
-X_train, X_test = X[train_index], X[test_index]
-y_train, y_test = y[train_index], y[test_index]
-
-# Initialize a classifier with tuning and fit
-clf_no_tuning = TabPFNClassifier(eval_metric="f1")
-clf_no_tuning.fit(X_train, y_train)
-
-# Predict F1 score without tuning
-predictions = clf_no_tuning.predict(X_test)
-print(f"F1 Score without tuning: {f1_score(y_test, predictions):.3f}")
-
-# Initialize a classifier with tuning and fit
-clf_with_tuning = TabPFNClassifier(
-    eval_metric="f1",
-    tuning_config={"tune_decision_thresholds": True},
-)
-# This will tune the temperature and decision thresholds
-clf_with_tuning.fit(X_train, y_train)
-
-# Predict F1 score with tuning
-predictions = clf_with_tuning.predict(X_test)
-print(f"F1 Score with tuning: {f1_score(y_test, predictions):.3f}")
diff --git a/pyproject.toml b/pyproject.toml
index 5446829..ab3bcee 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -4,34 +4,21 @@ build-backend = "setuptools.build_meta"
 
 [project]
 name = "tabpfn"
-version = "6.0.6"
+version = "2.1.1"
 dependencies = [
   "torch>=2.1,<3",
-  "numpy>=1.21.6,<3",
-  "scikit-learn>=1.2.0,<1.8",
+  "scikit-learn>=1.2.0,<1.7",
   "typing_extensions>=4.12.0",
   "scipy>=1.11.1,<2",
   "pandas>=1.4.0,<3",
   "einops>=0.2.0,<0.9",
-  "huggingface-hub>=0.19.0,<2",
+  "huggingface-hub>=0.0.1,<1",
   "pydantic>=2.8.0",
   "pydantic-settings>=2.10.1",
   # eval-type-backport is required on Python 3.9 to enable support for "X | Y" notation
   # for union types in Pydantic.
   # Once Python 3.10 is the minimum version, this can be removed.
   "eval-type-backport>=0.2.2",
-  "joblib>=1.2.0",
-  "tabpfn-common-utils[telemetry-interactive]>=0.2.7",
-  # pyobjc-framework-Metal is required to determine the available memory for MPS
-  # devices for PyTorch <2.5, so we only need it on MacOS.
-  # Once the minimum PyTorch >= 2.5, this can be removed.
-  "pyobjc-framework-Metal; sys_platform == 'darwin' and python_version > '3.9'",
-  # Special handling for pyobjc-framework-Metal/pyobjc-core on Python 3.9 since
-  # they stopped building packages for 3.9 starting with 12.0 and for some
-  # reason pinning only the direct dependency isn't enough.
-  "pyobjc-framework-Metal<13.0; sys_platform == 'darwin' and python_version == '3.9'",
-  "pyobjc-core<12.0; sys_platform == 'darwin' and python_version == '3.9'",
-  "kditransform>=1.2",
 ]
 requires-python = ">=3.9"
 authors = [
@@ -47,14 +34,12 @@ authors = [
   { name = "Eddie Bergman" },
   # Prior Labs Contributors
   { name = "Leo Grinsztajn" },
-  { name = "Felix Jabloski" },
-  { name = "Klemens Fl├╢ge" },
-  { name = "Oscar Key" },
-  { name = "Felix Birkel" },
-  { name = "Philipp Jund" },
-  { name = "Brendan Roof" },
-  { name = "Dominik Safaric" },
-  { name = "Benjamin Jaeger" }
+  { name = "Felix Jabloski"},
+  { name = "Klemens Fl├╢ge"},
+  { name = "Oscar Key"},
+  { name = "Felix Birkel"},
+  { name = "Philipp Jund"},
+
 ]
 
 readme = "README.md"
@@ -84,42 +69,27 @@ source = "https://github.com/priorlabs/tabpfn"
 [project.optional-dependencies]
 dev = [
   # Lint/format
-  "pre-commit>=4.3.0",
-  "ruff==0.14.0", # This version must be the same as in .pre-commit-config.yaml
-  "mypy==1.18.2", # This version must be the same as in .pre-commit-config.yaml
+  "pre-commit",
+  "ruff==0.8.6", # This version must be the same as in .pre-commit-config.yaml
+  "mypy==1.17.0", # This version must be the same as in .pre-commit-config.yaml
   # Test
-  "pytest>=8.4.2",
-  "pytest-xdist>=3.8.0",
-  "pytest-mock>=3.14.1",
-  "onnx>=1.19.0", # required for onnx export tests
-  "psutil>=7.1.0", # required for testing internal memory tool on windows
+  "pytest",
+  "pytest-xdist",
+  "onnx", # required for onnx export tests
+  "psutil", # required for testing internal memory tool on windows
   # Docs
-  "mkdocs>=1.6.1",
-  "mkdocs-material>=9.6.21",
-  "mkdocs-autorefs>=1.4.3",
-  "mkdocs-gen-files>=0.5.0",
-  "mkdocs-literate-nav>=0.6.2",
-  "mkdocs-glightbox>=0.5.1",
-  "mkdocstrings[python]>=0.30.1",
-  "markdown-exec[ansi]>=1.11.0",
-  "mike>=2.1.3",
-  # We use Ruff for formatting but this allows mkdocstrings to format signatures in the
-  # docs.
-  "black>=25.9.0",
-]
-# The minimum subset of the dev dependencies required to run the tests on the CI.
-# The idea is to be as close to the deployment environment as possible.
-ci = [
-  "licensecheck>=2025.1.0",
-  "onnx>=1.19.0",
-  "psutil>=7.1.0",
-  "pytest-mock>=3.14.1",
-  "pytest>=8.4.2",
+  "mkdocs",
+  "mkdocs-material",
+  "mkdocs-autorefs",
+  "mkdocs-gen-files",
+  "mkdocs-literate-nav",
+  "mkdocs-glightbox",
+  "mkdocstrings[python]",
+  "markdown-exec[ansi]",
+  "mike",
+  "black",  # This allows mkdocstrings to format signatures in the docs
 ]
 
-[tool.setuptools.package-data]
-"tabpfn.architectures.base" = ["tabpfn_col_embedding.pt"]
-
 [tool.pytest.ini_options]
 testpaths = ["tests"]  # Where the tests are located
 minversion = "8.0"
@@ -136,10 +106,6 @@ line-length = 88
 output-format = "full"
 src = ["src", "tests", "examples"]
 
-extend-exclude = [
-  "**/*.ipynb"
-]
-
 [tool.ruff.lint]
 # Extend what ruff is allowed to fix, even it it may break
 # This is okay given we use it all the time and it ensures
@@ -152,7 +118,7 @@ dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"
 
 select = [
   "A",
-  "ANN", # type annotations
+  # "ANN", # Handled by mypy
   "ARG",
   "B",
   "BLE",
@@ -219,10 +185,7 @@ ignore = [
   "N803",    # Argument name `X` should be lowercase
   "N802",    # Function name should be lowercase
   "COM812",  # Trailing comma missing (conflicts with formatter)
-  "ANN002",  # No type annotation for *args is okay.
-  "ANN003",  # No type annotation for *kwargs is okay.
-  "ANN204",  # No type annotation for __init__ is okay.
-  "ANN401"   # We do allow Any annotations, but use responsibly.
+  # These tend to be lighweight and confuse pyright
 ]
 
 exclude = [
@@ -246,14 +209,10 @@ exclude = [
   "node_modules",
   "venv",
   "docs",
-  "**/*.ipynb",
 ]
 
+# Exclude a variety of commonly ignored directories.
 [tool.ruff.lint.per-file-ignores]
-# These files are copied without changes from extenal repositories, so ignore all rules.
-"src/tabpfn/misc/debug_versions.py" = ["ALL"]
-"src/tabpfn/misc/_sklearn_compat.py" = ["ALL"]
-
 "tests/*.py" = [
   "S101",
   "D101",
@@ -293,15 +252,19 @@ exclude = [
   "FBT002",
   "A001",
 ]
-"src/tabpfn/architectures/base/preprocessing.py" = [
-  "ANN"
-]
 "src/tabpfn/model_loading.py" = [
   "C901"
 ]
 "src/tabpfn/*.py" = [
   "D107",
 ]
+"examples/notebooks/TabPFN_Demo_Local.ipynb" = [
+    "F401",  # Unused import
+    "A004",  # Shadowing builtin
+    "PD901", # Generic variable name `df`
+    "NPY002",# Legacy np.random call
+    "ARG001",# Unused function argument
+]
 
 
 [tool.ruff.lint.isort]
@@ -403,27 +366,3 @@ reportPrivateImportUsage = false
 reportUnnecessaryComparison = false
 reportConstantRedefinition = false
 reportUntypedFunctionDecorator = false
-
-[tool.licensecheck]
-# Acceptable licenses
-only_licenses = [
-    "APACHE",
-    "MIT",
-    "BSD",
-    "ISC",
-    "PYTHON",
-    "UNLICENSE",
-]
-
-# Packages that we don't consider
-ignore_packages = [
-    # Uses MPL, but acceptable because we don't modify it. https://github.com/certifi/python-certifi/blob/fb14ac49a976b1695d84b1ac1307276a20b3aac9/LICENSE
-    "certifi",
-    # Appears to be MIT, but the tool doesn't know. https://github.com/arogozhnikov/einops/blob/361b11e87da94ead4bd09de636c5dbed73e0e3e0/LICENSE
-    "einops",
-    # Is Apache Licensed since 1.2 but the tool doesn't know. https://github.com/calvinmccarter/kditransform/blob/5ee7cfad665bb1078211c0becad8fbd31e78429d/LICENSE
-    "kditransform",
-    "nvidia*",
-    # Our packages
-    "tabpfn*",
-]
diff --git a/scripts/download_all_models.py b/scripts/download_all_models.py
index 2e814b2..90d28a3 100644
--- a/scripts/download_all_models.py
+++ b/scripts/download_all_models.py
@@ -4,9 +4,10 @@ from __future__ import annotations
 
 import argparse
 import logging
+import sys
 from pathlib import Path
 
-from tabpfn.model_loading import download_all_models, get_cache_dir
+from tabpfn.model_loading import _user_cache_dir, download_all_models
 
 
 def main() -> None:
@@ -28,7 +29,9 @@ def main() -> None:
     logger = logging.getLogger(__name__)
 
     # Determine cache directory
-    cache_dir = args.cache_dir or get_cache_dir()
+    cache_dir = args.cache_dir or _user_cache_dir(
+        platform=sys.platform, appname="tabpfn"
+    )
     cache_dir.mkdir(parents=True, exist_ok=True)
 
     logger.info(f"Downloading all models to {cache_dir}")
diff --git a/scripts/generate_dependencies.py b/scripts/generate_dependencies.py
new file mode 100644
index 0000000..e1acedb
--- /dev/null
+++ b/scripts/generate_dependencies.py
@@ -0,0 +1,90 @@
+"""Generate a requirements.txt file from pyproject.toml dependencies.
+
+This script can operate in two modes:
+1. 'min': Extracts minimum versions (>=) and pins them with '=='.
+2. 'max': Extracts maximum versions (<) or leaves them unpinned.
+"""
+
+from __future__ import annotations
+
+import argparse
+import re
+from pathlib import Path
+
+
+def parse_dependency_lines(content: str) -> list[str]:
+    """Finds and cleans the dependency lines from the pyproject.toml content."""
+    deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
+    if not deps_match:
+        return []
+
+    deps_lines = deps_match.group(1).strip().split("\n")
+
+    cleaned_deps = []
+    for line in deps_lines:
+        # Assign the stripped line to a new variable to avoid the linter warning.
+        stripped_line = line.strip()
+        # Skip empty lines or comments
+        if not stripped_line or stripped_line.startswith("#"):
+            continue
+        # Clean the line by removing an optional trailing comma, then stripping quotes.
+        clean_dep = stripped_line.rstrip(",").strip("'\"")
+        cleaned_deps.append(clean_dep)
+
+    return cleaned_deps
+
+
+def main() -> None:
+    """Main function to parse arguments and generate the requirements file."""
+    parser = argparse.ArgumentParser(
+        description="Generate requirements.txt from pyproject.toml.",
+        formatter_class=argparse.RawTextHelpFormatter,
+    )
+    parser.add_argument(
+        "mode",
+        choices=["minimum", "maximum"],
+        help="The type of requirements to generate:\n"
+        "'minimum' - for minimum versions (e.g., 'package==1.2.3')\n"
+        "'maximum' - for maximum/unpinned versions (e.g., 'package<2.0' or 'package')",
+    )
+    args = parser.parse_args()
+
+    try:
+        content = Path("pyproject.toml").read_text()
+    except FileNotFoundError:
+        return
+
+    # 1. Shared parsing logic
+    deps = parse_dependency_lines(content)
+    output_reqs = []
+
+    # 2. Mode-specific processing logic
+    if args.mode == "maximum":
+        for dep in deps:
+            # Check for maximum version constraint
+            max_version_match = re.search(r'([^>=<\s]+).*?<\s*([^,\s"\']+)', dep)
+            if max_version_match:
+                package, max_ver = max_version_match.groups()
+                output_reqs.append(f"{package}<{max_ver}")
+            else:
+                # If no max version, just use the package name
+                package_match = re.match(r"([^>=<\s]+)", dep)
+                if package_match:
+                    output_reqs.append(package_match.group(1))
+
+    elif args.mode == "minimum":
+        for dep in deps:
+            # Check for minimum version constraint
+            match = re.match(r'([^>=<\s]+)\s*>=\s*([^,\s"\']+)', dep)
+            if match:
+                package, min_ver = match.groups()
+                output_reqs.append(f"{package}=={min_ver}")
+
+    # 3. Shared writing logic
+    output_filename = "requirements.txt"
+    with Path(output_filename).open("w") as f:
+        f.write("\n".join(output_reqs))
+
+
+if __name__ == "__main__":
+    main()
diff --git a/src/tabpfn/__init__.py b/src/tabpfn/__init__.py
index 5615db0..6377041 100644
--- a/src/tabpfn/__init__.py
+++ b/src/tabpfn/__init__.py
@@ -2,10 +2,7 @@ from importlib.metadata import version
 
 from tabpfn.classifier import TabPFNClassifier
 from tabpfn.misc.debug_versions import display_debug_info
-from tabpfn.model_loading import (
-    load_fitted_tabpfn_model,
-    save_fitted_tabpfn_model,
-)
+from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
 from tabpfn.regressor import TabPFNRegressor
 
 try:
diff --git a/src/tabpfn/architectures/base/__init__.py b/src/tabpfn/architectures/base/__init__.py
index 4966b34..96d8a2f 100644
--- a/src/tabpfn/architectures/base/__init__.py
+++ b/src/tabpfn/architectures/base/__init__.py
@@ -6,15 +6,14 @@ architectures can import components from here to reuse them, and over time we sh
 refactor this architecture to improve reusability.
 """
 
-from __future__ import annotations
+from typing import Any
 
-from typing import TYPE_CHECKING, Any, Literal
+from torch import nn
 
 from tabpfn.architectures.base.config import ModelConfig
 from tabpfn.architectures.base.encoders import (
     InputNormalizationEncoderStep,
     LinearInputEncoderStep,
-    MLPInputEncoderStep,
     MulticlassClassificationTargetEncoder,
     NanHandlingEncoderStep,
     RemoveDuplicateFeaturesEncoderStep,
@@ -24,11 +23,7 @@ from tabpfn.architectures.base.encoders import (
     VariableNumFeaturesEncoderStep,
 )
 from tabpfn.architectures.base.transformer import PerFeatureTransformer
-
-if TYPE_CHECKING:
-    from torch import nn
-
-    from tabpfn.architectures.interface import ArchitectureConfig
+from tabpfn.architectures.interface import ArchitectureConfig
 
 
 def parse_config(config: dict[str, Any]) -> tuple[ArchitectureConfig, dict[str, Any]]:
@@ -91,9 +86,6 @@ def get_architecture(
             remove_outliers=config.remove_outliers,
             normalize_by_used_features=config.normalize_by_used_features,
             encoder_use_bias=config.encoder_use_bias,
-            encoder_type=config.encoder_type,
-            encoder_mlp_hidden_dim=config.encoder_mlp_hidden_dim,
-            encoder_mlp_num_layers=config.encoder_mlp_num_layers,
         ),
         y_encoder=get_y_encoder(
             num_inputs=1,
@@ -124,9 +116,6 @@ def get_encoder(  # noqa: PLR0913
     remove_outliers: bool,
     normalize_by_used_features: bool,
     encoder_use_bias: bool,
-    encoder_type: Literal["linear", "mlp"] = "linear",
-    encoder_mlp_hidden_dim: int | None = None,
-    encoder_mlp_num_layers: int = 2,
 ) -> nn.Module:
     inputs_to_merge = {"main": {"dim": num_features}}
 
@@ -167,34 +156,15 @@ def get_encoder(  # noqa: PLR0913
         ),
     ]
 
-    num_input_features = sum(i["dim"] for i in inputs_to_merge.values())
-    if encoder_type == "mlp":
-        encoder_steps += [
-            MLPInputEncoderStep(
-                num_features=num_input_features,
-                emsize=embedding_size,
-                hidden_dim=encoder_mlp_hidden_dim,
-                activation="gelu",
-                num_layers=encoder_mlp_num_layers,
-                bias=encoder_use_bias,
-                in_keys=tuple(inputs_to_merge),
-                out_keys=("output",),
-            ),
-        ]
-    elif encoder_type == "linear":
-        encoder_steps += [
-            LinearInputEncoderStep(
-                num_features=num_input_features,
-                emsize=embedding_size,
-                bias=encoder_use_bias,
-                in_keys=tuple(inputs_to_merge),
-                out_keys=("output",),
-            ),
-        ]
-    else:
-        raise ValueError(
-            f"Invalid encoder type: {encoder_type} (expected 'linear' or 'mlp')"
-        )
+    encoder_steps += [
+        LinearInputEncoderStep(
+            num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
+            emsize=embedding_size,
+            bias=encoder_use_bias,
+            in_keys=tuple(inputs_to_merge),
+            out_keys=("output",),
+        ),
+    ]
 
     return SequentialEncoder(*encoder_steps, output_key="output")
 
diff --git a/src/tabpfn/architectures/base/attention/full_attention.py b/src/tabpfn/architectures/base/attention/full_attention.py
index 8183ff3..00896a8 100644
--- a/src/tabpfn/architectures/base/attention/full_attention.py
+++ b/src/tabpfn/architectures/base/attention/full_attention.py
@@ -3,6 +3,7 @@
 
 from __future__ import annotations
 
+import contextlib
 import math
 from functools import partial
 from typing import TYPE_CHECKING
@@ -17,37 +18,16 @@ from tabpfn.architectures.base.memory import support_save_peak_mem_factor
 if TYPE_CHECKING:
     from tabpfn.architectures.base.config import ModelConfig
 
-TORCH_VERSION = torch.__version__.split(".")
+try:
+    from flash_attn.flash_attn_interface import (
+        flash_attn_unpadded_func,
+        flash_attn_unpadded_kvpacked_func,
+        flash_attn_unpadded_qkvpacked_func,
+    )
 
-TORCH_2_ATTENTION_POSSIBLE = int(TORCH_VERSION[0]) >= 2
-
-
-def _gqa_is_supported() -> bool:
-    """Check if PyTorch's scaled_dot_product_attention supports enable_gqa parameter.
-
-    This checks whether torch.nn.functional.scaled_dot_product_attention has a
-    kwarg enable_gqa and if we have sufficient NVIDIA compute capability.
-    PyTorch 2.5+ includes enable_gqa support.
-    """
-    if not TORCH_2_ATTENTION_POSSIBLE or not torch.cuda.is_available():
-        return False
-
-    # Check if PyTorch version is 2.5 or higher for enable_gqa support
-    torch_major, torch_minor = int(TORCH_VERSION[0]), int(TORCH_VERSION[1])
-    has_enable_gqa = torch_major > 2 or (torch_major == 2 and torch_minor >= 5)
-
-    if not has_enable_gqa:
-        return False
-
-    # Check compute capability only if CUDA is available
-    # We need compute capability >= 8.0 for efficient GQA
-    device = torch.cuda.current_device()
-    nvidia_compute_capability = torch.cuda.get_device_capability(device)
-    return nvidia_compute_capability[0] >= 8
-
-
-# Cache the GQA support check at module level
-USE_TORCH_2_GQA = _gqa_is_supported()
+    HAVE_FLASH_ATTN = True
+except (ModuleNotFoundError, ImportError):
+    HAVE_FLASH_ATTN = False
 
 
 class MultiHeadAttention(Attention):
@@ -134,7 +114,7 @@ class MultiHeadAttention(Attention):
         def assert_tensor_shape(
             tensor: torch.Tensor | None,
             expected_shape: list[int | None],
-        ) -> None:
+        ):
             if tensor is None:
                 return
             actual_shape = tensor.size()
@@ -302,13 +282,13 @@ class MultiHeadAttention(Attention):
         Else, keys and values are attained by applying the respective linear
         transformations to 'x' (self attention).
         """
-        assert not (cache_kv and use_cached_kv), (
-            "Cannot cache and use cached keys and values at the same time."
-        )
+        assert not (
+            cache_kv and use_cached_kv
+        ), "Cannot cache and use cached keys and values at the same time."
 
-        assert not x.requires_grad or (not self.has_cached_kv and not cache_kv), (
-            "Saving keys and values is only supported during inference."
-        )
+        assert not x.requires_grad or (
+            not self.has_cached_kv and not cache_kv
+        ), "Saving keys and values is only supported during inference."
         x, x_kv, x_shape_after_transpose = self._rearrange_inputs_to_flat_batch(x, x_kv)
 
         nhead_kv = 1 if reuse_first_head_kv else self._nhead_kv
@@ -384,9 +364,9 @@ class MultiHeadAttention(Attention):
         torch.Tensor | None,
         torch.Tensor | None,
     ]:
-        assert not (cache_kv and use_cached_kv), (
-            "You cannot both cache new KV and use the cached KV at once."
-        )
+        assert not (
+            cache_kv and use_cached_kv
+        ), "You cannot both cache new KV and use the cached KV at once."
         if reuse_first_head_kv:
             assert x is not x_kv, (
                 "x and x_kv must be different tensors. That means reuse_first_head_kv"
@@ -397,9 +377,9 @@ class MultiHeadAttention(Attention):
 
         k = v = kv = None
         if use_cached_kv:
-            assert self.has_cached_kv, (
-                "You try to use cached keys and values but the cache is empty."
-            )
+            assert (
+                self.has_cached_kv
+            ), "You try to use cached keys and values but the cache is empty."
             k = k_cache
             v = v_cache
             kv = kv_cache
@@ -418,18 +398,7 @@ class MultiHeadAttention(Attention):
             and k is None
             and v is None
         ):
-            # A faster version of
-            # qkv = torch.einsum("... s, j h d s -> ... j h d", x, self._w_qkv)
-            batch_shape = x.shape[:-1]  # [..., seq_len]
-            j, nhead, d_k, input_size = self._w_qkv.shape
-
-            # [j, nhead, d_k, input_size] -> [j * nhead * d_k, input_size]
-            w_flat = self._w_qkv.reshape(-1, input_size)
-
-            qkv_flat = torch.matmul(x, w_flat.T)
-
-            # Reshape back to desired format: [..., seq_len, j, nhead, d_k]
-            qkv = qkv_flat.reshape(*batch_shape, j, nhead, d_k)
+            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self._w_qkv)
             q = None
         else:
             qkv = None
@@ -536,9 +505,6 @@ class MultiHeadAttention(Attention):
         kv: torch.Tensor,
         share_kv_across_n_heads: int,
     ) -> torch.Tensor:
-        if share_kv_across_n_heads == 1:
-            return kv
-
         nhead, d = kv.shape[-2:]
         kv = kv[..., None, :].expand(
             *([-1] * (kv.dim() - 1)),
@@ -548,54 +514,7 @@ class MultiHeadAttention(Attention):
         return kv.reshape(*kv.shape[:-3], nhead * share_kv_across_n_heads, d)
 
     @staticmethod
-    def scaled_dot_product_attention_chunked(
-        q: torch.Tensor,
-        k: torch.Tensor,
-        v: torch.Tensor,
-        dropout_p: float | None = None,
-        max_batch_size: int = 65_000,
-        **extra_inputs,
-    ) -> torch.Tensor:
-        """Scaled dot product attention with automatic chunking to handle
-        batch size limitations when batch size is larger than 65_535.
-        This is a workaround for the issue: https://github.com/pytorch/pytorch/issues/142228.
-
-        Args:
-            q: Query tensor
-            k: Key tensor
-            v: Value tensor
-            dropout_p: Dropout probability
-            max_batch_size: Maximum batch size for CUDA kernels (default 65_000)
-            extra_inputs: Additional arguments for scaled_dot_product_attention
-
-        Returns:
-            Attention output with same shape as input q
-        """
-        batch_size = q.shape[0]
-        output_chunks = []
-
-        for start_idx in range(0, batch_size, max_batch_size):
-            end_idx = min(start_idx + max_batch_size, batch_size)
-
-            q_chunk = q[start_idx:end_idx]
-            k_chunk = k[start_idx:end_idx]
-            v_chunk = v[start_idx:end_idx]
-
-            chunk_output = torch.nn.functional.scaled_dot_product_attention(
-                q_chunk,
-                k_chunk,
-                v_chunk,
-                dropout_p=dropout_p,
-                **extra_inputs,
-            )
-
-            output_chunks.append(chunk_output)
-
-        # Concatenate results along batch dimension
-        return torch.cat(output_chunks, dim=0)
-
-    @staticmethod
-    def compute_attention_heads(
+    def compute_attention_heads(  # noqa: C901, PLR0912
         q: torch.Tensor | None,
         k: torch.Tensor | None,
         v: torch.Tensor | None,
@@ -616,24 +535,107 @@ class MultiHeadAttention(Attention):
         assert q is not None
         assert k is not None
         assert v is not None
-
         batch_size, seqlen_q, nhead, d_k = q.shape
-        _, _seqlen_kv, nhead_kv, d_v = v.shape
+        _, seqlen_kv, nhead_kv, d_v = v.shape
         share_kv_across_n_heads = nhead // nhead_kv
         if dropout_p is None:
             dropout_p = 0.0  # TODO: necessary?
 
+        use_flash_attention = (
+            HAVE_FLASH_ATTN
+            and torch.cuda.is_available()
+            and q.dtype == k.dtype == v.dtype == torch.float16
+        )
+
+        # this string comparison is reliable, as it does not compare to a subversion
+        TORCH_2_ATTENTION_POSSIBLE = (
+            torch.__version__ >= "2" and torch.cuda.is_available()
+        )
+        USE_TORCH_2_GQA = False
         if TORCH_2_ATTENTION_POSSIBLE:
-            extra_inputs = {}
-            if softmax_scale is not None:
-                extra_inputs["scale"] = (
-                    softmax_scale  # defaults to 1/sqrt(d_k) if None or not provided
+            # check whether torch.nn.functional.scaled_dot_product_attention has a
+            # kwarg enable_gqa
+            # Check if enable_gqa is supported by trying to call the function with
+            # the parameter
+            with contextlib.suppress(TypeError, RuntimeError):
+                _ = torch.nn.functional.scaled_dot_product_attention(
+                    torch.empty(1, 1, 1, 1),
+                    torch.empty(1, 1, 1, 1),
+                    torch.empty(1, 1, 1, 1),
+                    enable_gqa=True,
                 )
 
-            # Check if we should use PyTorch 2.0's GQA support
-            if USE_TORCH_2_GQA:
-                extra_inputs["enable_gqa"] = True
+            # if torch.cuda.is_available():
+            #     device = torch.cuda.current_device()
+            #     capability = torch.cuda.get_device_capability(device)
+            #     nvidia_compute_capability = f"{capability[0]}.{capability[1]}"
+            # else:
+            #     nvidia_compute_capability = None
+            # USE_TORCH_2_GQA = nvidia_compute_capability >= "8" and TORCH_2_SUPPORTS_GQ
+            # The code above hangs on multi-gpu settings,
+            # so we use a temporary solution:
+            USE_TORCH_2_GQA = True  # TODO
+            # TODO: add logging for something like this
+            # if use_flash_attention and USE_TORCH_2_GQA:
+            # print("Using FlashAttention might be slower than torch's implementation,
+            # try setting
+            # `tabpfn.architectures.base.multi_head_attention.HAVE_FLASH_ATTN=False`.")
+
+            # print(f"USE_TORCH_2_GQA: {USE_TORCH_2_GQA}, nvidia_compute_capability:
+            # {nvidia_compute_capability}, TORCH_2_SUPPORTS_GQ: {TORCH_2_SUPPORTS_GQ}")
+
+        if use_flash_attention:
+
+            def get_seqlen_cumsums(
+                batch_size: int,
+                seqlen: int,
+                device: torch.device,
+            ) -> torch.Tensor:
+                return torch.arange(
+                    0,
+                    (batch_size + 1) * seqlen,
+                    step=seqlen,
+                    dtype=torch.int32,
+                    device=device,
+                )
+
+            if qkv is not None:
+                attention_head_outputs = flash_attn_unpadded_qkvpacked_func(  # type: ignore
+                    qkv.reshape(batch_size * seqlen_q, 3, nhead, d_k),
+                    get_seqlen_cumsums(batch_size, seqlen_q, qkv.device),
+                    seqlen_q,
+                    dropout_p=dropout_p,
+                    softmax_scale=softmax_scale,  # defaults to 1/sqrt(d_k) if None
+                    causal=False,
+                    return_attn_probs=False,
+                    deterministic=False,
+                )
+            elif kv is not None:
+                kv = MultiHeadAttention.broadcast_kv_across_heads(
+                    kv,
+                    share_kv_across_n_heads,
+                )
+                attention_head_outputs = flash_attn_unpadded_kvpacked_func(  # type: ignore
+                    q.reshape(batch_size * seqlen_q, nhead, d_k),
+                    kv.reshape(batch_size * seqlen_kv, 2, nhead, d_k),
+                    get_seqlen_cumsums(batch_size, seqlen_q, q.device),
+                    get_seqlen_cumsums(batch_size, seqlen_kv, kv.device),
+                    seqlen_q,
+                    seqlen_kv,
+                    dropout_p=dropout_p,
+                    causal=False,
+                    return_attn_probs=False,
+                    deterministic=False,
+                )
             else:
+                assert d_k <= d_v, (
+                    "This requirement is here for safety but not strictly necessary."
+                    "Needs testing/coding to remove."
+                )
+                if d_k < d_v:
+                    k = torch.nn.functional.pad(k, d_v - d_k)
+                    q = torch.nn.functional.pad(v, d_v - d_k)
+                    d_k_ = d_v
                 k = MultiHeadAttention.broadcast_kv_across_heads(
                     k,
                     share_kv_across_n_heads,
@@ -642,18 +644,45 @@ class MultiHeadAttention(Attention):
                     v,
                     share_kv_across_n_heads,
                 )
-
-            attention_head_outputs = (
-                MultiHeadAttention.scaled_dot_product_attention_chunked(
-                    q.transpose(1, 2),
-                    k.transpose(1, 2),
-                    v.transpose(1, 2),
+                attention_head_outputs = flash_attn_unpadded_func(  # type: ignore
+                    q.reshape(batch_size * seqlen_q, nhead, d_k_),  # type: ignore
+                    k.reshape(batch_size * seqlen_kv, nhead, d_k_),  # type: ignore
+                    v.reshape(batch_size * seqlen_kv, nhead, d_v),
+                    get_seqlen_cumsums(batch_size, seqlen_q, q.device),
+                    get_seqlen_cumsums(batch_size, seqlen_kv, k.device),
+                    seqlen_q,
+                    seqlen_kv,
                     dropout_p=dropout_p,
-                    **extra_inputs,
+                    softmax_scale=softmax_scale,
+                    causal=False,
+                    return_attn_probs=False,
+                    deterministic=False,
+                )
+        elif TORCH_2_ATTENTION_POSSIBLE:
+            extra_inputs = {}
+            if softmax_scale is not None:
+                extra_inputs["scale"] = (
+                    softmax_scale  # defaults to 1/sqrt(d_k) if None or not provided
                 )
+            if not USE_TORCH_2_GQA:
+                k = MultiHeadAttention.broadcast_kv_across_heads(
+                    k,
+                    share_kv_across_n_heads,
+                )
+                v = MultiHeadAttention.broadcast_kv_across_heads(
+                    v,
+                    share_kv_across_n_heads,
+                )
+            else:
+                extra_inputs["enable_gqa"] = True
+            attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(
+                q.transpose(1, 2),
+                k.transpose(1, 2),
+                v.transpose(1, 2),
+                dropout_p=dropout_p,
+                **extra_inputs,
             )
             attention_head_outputs = attention_head_outputs.transpose(1, 2)
-
         else:
             k = MultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
             v = MultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)
diff --git a/src/tabpfn/architectures/base/bar_distribution.py b/src/tabpfn/architectures/base/bar_distribution.py
index 0ecc45e..4062143 100644
--- a/src/tabpfn/architectures/base/bar_distribution.py
+++ b/src/tabpfn/architectures/base/bar_distribution.py
@@ -48,10 +48,6 @@ class BarDistribution(nn.Module):
         self.ignore_nan_targets = ignore_nan_targets
         self.to(borders.device)
 
-    def has_equal_borders(self, other: BarDistribution) -> bool:
-        """Check if two BarDistributions have equal borders."""
-        return torch.equal(self.borders, other.borders)  # pyright: ignore[reportArgumentType]
-
     @property
     def bucket_widths(self) -> torch.Tensor:
         return self.borders[1:] - self.borders[:-1]
@@ -74,11 +70,11 @@ class BarDistribution(nn.Module):
         """
         if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
             # bring new borders to the same dim as logits up to the last dim
-            ys = ys.repeat((*logits.shape[:-1], 1))
+            ys = ys.repeat(logits.shape[:-1] + (1,))
         else:
-            assert ys.shape[:-1] == logits.shape[:-1], (
-                f"ys.shape: {ys.shape} logits.shape: {logits.shape}"
-            )
+            assert (
+                ys.shape[:-1] == logits.shape[:-1]
+            ), f"ys.shape: {ys.shape} logits.shape: {logits.shape}"
         probs = torch.softmax(logits, dim=-1)
         buckets_of_ys = self.map_to_bucket_idx(ys).clamp(0, self.num_bars - 1)
 
@@ -193,9 +189,9 @@ class BarDistribution(nn.Module):
         ignore_loss_mask = self.ignore_init(y)
         target_sample = self.map_to_bucket_idx(y)
         assert (target_sample >= 0).all()
-        assert (target_sample < self.num_bars).all(), (
-            f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
-        )
+        assert (
+            target_sample < self.num_bars
+        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
 
         last_dim = logits.shape[-1]
         assert last_dim == self.num_bars, f"{last_dim} v {self.num_bars}"
@@ -419,8 +415,7 @@ class BarDistribution(nn.Module):
         **kwargs: Any,
     ) -> plt.Axes:
         """Plots the distribution."""
-        # Local import because matplotlib is an optional dependency.
-        import matplotlib.pyplot as plt  # noqa: PLC0415
+        import matplotlib.pyplot as plt
 
         logits = logits.squeeze()
         assert logits.dim() == 1, "logits should be 1d, at least after squeezing."
@@ -466,9 +461,9 @@ class FullSupportBarDistribution(BarDistribution):
 
     def assert_support(self, *, allow_zero_bucket_left: bool = False) -> None:
         if allow_zero_bucket_left:
-            assert self.bucket_widths[-1] > 0, (
-                f"Half Normal weight must be > 0 (got -1:{self.bucket_widths[-1]})."
-            )
+            assert (
+                self.bucket_widths[-1] > 0
+            ), f"Half Normal weight must be > 0 (got -1:{self.bucket_widths[-1]})."
             # This fixes the distribution if the half normal at zero is width zero
             if self.bucket_widths[0] == 0:
                 self.borders[0] = self.borders[0] - 1
@@ -509,13 +504,13 @@ class FullSupportBarDistribution(BarDistribution):
         target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
         target_sample.clamp_(0, self.num_bars - 1)
 
-        assert logits.shape[-1] == self.num_bars, (
-            f"{logits.shape[-1]} vs {self.num_bars}"
-        )
+        assert (
+            logits.shape[-1] == self.num_bars
+        ), f"{logits.shape[-1]} vs {self.num_bars}"
         assert (target_sample >= 0).all()
-        assert (target_sample < self.num_bars).all(), (
-            f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
-        )
+        assert (
+            target_sample < self.num_bars
+        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
         last_dim = logits.shape[-1]
         assert last_dim == self.num_bars, f"{last_dim} vs {self.num_bars}"
         # ignore all position with nan values
@@ -548,9 +543,9 @@ class FullSupportBarDistribution(BarDistribution):
         nll_loss = -log_probs
 
         if mean_prediction_logits is not None:  # TO BE REMOVED AFTER BO PAPER IS DONE
-            assert not ignore_loss_mask.any(), (
-                "Ignoring examples is not implemented with mean pred."
-            )
+            assert (
+                not ignore_loss_mask.any()
+            ), "Ignoring examples is not implemented with mean pred."
             if not torch.is_grad_enabled():
                 pass
             nll_loss = torch.cat(
@@ -787,16 +782,16 @@ def get_bucket_limits(
             If set, the bucket limits are widened by this factor.
             This allows to have a slightly larger range than the actual data.
     """
-    assert (ys is None) != (full_range is None), (
-        "Either full_range or ys must be passed."
-    )
+    assert (ys is None) != (
+        full_range is None
+    ), "Either full_range or ys must be passed."
 
     if ys is not None:
         ys = ys.flatten()
         ys = ys[~torch.isnan(ys)]
-        assert len(ys) > num_outputs, (
-            f"Number of ys :{len(ys)} must be larger than num_outputs: {num_outputs}"
-        )
+        assert (
+            len(ys) > num_outputs
+        ), f"Number of ys :{len(ys)} must be larger than num_outputs: {num_outputs}"
         if len(ys) % num_outputs:
             ys = ys[: -(len(ys) % num_outputs)]
         ys_per_bucket = len(ys) // num_outputs
@@ -807,7 +802,7 @@ def get_bucket_limits(
             assert full_range[1] >= ys.max()
             full_range = torch.tensor(full_range)  # type: ignore
 
-        ys_sorted, _ = ys.sort(0)  # type: ignore
+        ys_sorted, ys_order = ys.sort(0)  # type: ignore
         bucket_limits = (
             ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
             + ys_sorted[ys_per_bucket::ys_per_bucket]
diff --git a/src/tabpfn/architectures/base/config.py b/src/tabpfn/architectures/base/config.py
index 7ca489e..d850aba 100644
--- a/src/tabpfn/architectures/base/config.py
+++ b/src/tabpfn/architectures/base/config.py
@@ -8,7 +8,6 @@ from typing import Any, Literal, Optional
 from typing_extensions import Self
 
 import pydantic
-from pydantic import PositiveInt
 from pydantic.dataclasses import dataclass
 
 from tabpfn.architectures.interface import ArchitectureConfig
@@ -27,7 +26,7 @@ class ModelConfig(ArchitectureConfig):
     # ------ Actual variation across configs
     emsize: int = 192
     """The embedding dimension."""
-    features_per_group: PositiveInt = 2
+    features_per_group: Literal[1, 2] = 2
     """If > 1, the features will be grouped into groups of this size and the attention
     is across groups."""
     nhead: int = 6
@@ -39,14 +38,6 @@ class ModelConfig(ArchitectureConfig):
     # --- Constant across all configs and used
     dropout: float = 0.0
     encoder_use_bias: bool = False
-    encoder_type: Literal["linear", "mlp"] = "linear"
-    """Type of input encoder to use. Either "linear" for a simple linear layer or "mlp"
-    for a multi-layer perceptron."""
-    encoder_mlp_hidden_dim: int | None = 1024
-    """Hidden dimension for MLP encoder. If None, defaults to emsize. Only used when
-    encoder_type="mlp"."""
-    encoder_mlp_num_layers: int = 2
-    """Number of layers in the MLP encoder. Only used when encoder_type="mlp"."""
     feature_positional_embedding: FeaturePositionalEmbedding = "subspace"
     multiquery_item_attention: Literal[False] = False
     """When True, uses multiquery for attention between items."""
@@ -68,16 +59,14 @@ class ModelConfig(ArchitectureConfig):
     recompute_layer: bool = True
     """If True, enables activation checkpointing for each PerFeatureEncoderLayer in the
     encoder. This saves memory. recompute_attn is a related flag which checkpoints the
-    attention and mlp layers individually. Note that the forward pass takes an argument
-    `force_recompute_layer` which can be used to force recomputation of the layer."""
+    attention and mlp layers individually."""
     remove_empty_features: Literal[True] = True
     remove_outliers: Literal[False] = False
     use_separate_decoder: Literal[False] = False
     """If True, the decoder will be separate from the encoder."""
 
-    multiquery_item_attention_for_test_set: bool = True
-    """If True, uses multiquery attention on the test set.
-    For now, this must be False for bridge attention and True otherwise."""
+    multiquery_item_attention_for_test_set: Literal[True] = True
+    """If true, uses multiquery attention on the test set."""
 
     attention_init_gain: float = 1.0
     """The gain when initializing the attention parameters. If None, then 1.0 is
@@ -95,11 +84,6 @@ class ModelConfig(ArchitectureConfig):
     (though I'm not sure it makes a difference for a trained model).
     """
 
-    num_thinking_rows: int = 0
-    """If >0, then this number of "thinking rows" will be prepended to each dataset.
-    See tabpfn.architectures.base.AddThinkingTokens for an explanation.
-    """
-
     @classmethod
     def upgrade_config(cls, config: dict[str, Any]) -> dict[str, Any]:
         """Upgrade old configs to match the current config.
@@ -108,6 +92,7 @@ class ModelConfig(ArchitectureConfig):
         Raises a ValueError if the config is not compatible with the current code.
         """
         # The dates are to help us remove upgrades when they get very old.
+
         config = deepcopy(config)
 
         # Config changed on unknown date
diff --git a/src/tabpfn/architectures/base/encoders.py b/src/tabpfn/architectures/base/encoders.py
index fe07653..5932a8e 100644
--- a/src/tabpfn/architectures/base/encoders.py
+++ b/src/tabpfn/architectures/base/encoders.py
@@ -187,9 +187,9 @@ def normalize_data(
     Returns:
         The normalized data tensor, or a tuple containing the data and scaling factors.
     """
-    assert (mean is None) == (std is None), (
-        "Either both or none of mean and std must be given"
-    )
+    assert (mean is None) == (
+        std is None
+    ), "Either both or none of mean and std must be given"
     if mean is None:
         if normalize_positions is not None and normalize_positions > 0:
             mean = torch_nanmean(data[:normalize_positions], axis=0)  # type: ignore
@@ -233,26 +233,25 @@ def select_features(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
         The shape is (sequence_length, batch_size, total_features) if batch_size is greater than 1.
     """
     B, total_features = sel.shape
-
-    # Do nothing if we need to select all of the features
-    if torch.all(sel):
-        return x
+    sequence_length = x.shape[0]
 
     # If B == 1, we don't need to append zeros, as the number of features don't need to be fixed.
     if B == 1:
         return x[:, :, sel[0]]
 
-    new_x = x.detach().clone()
+    new_x = torch.zeros(
+        (sequence_length, B, total_features),
+        device=x.device,
+        dtype=x.dtype,
+    )
 
     # For each batch, compute the number of selected features.
     sel_counts = sel.sum(dim=-1)  # shape: (B,)
 
     for b in range(B):
         s = int(sel_counts[b])
-        if s != total_features:
-            if s > 0:
-                new_x[:, b, :s] = x[:, b, sel[b]]
-            new_x[:, b, s:] = 0
+        if s > 0:
+            new_x[:, b, :s] = x[:, b, sel[b]]
 
     return new_x
 
@@ -422,7 +421,7 @@ class SeqEncStep(nn.Module):
         *x: torch.Tensor,
         single_eval_pos: int | None = None,
         **kwargs: Any,
-    ) -> tuple[torch.Tensor | None, ...]:
+    ) -> tuple[torch.Tensor]:
         """Transform the data using the fitted encoder step.
 
         Args:
@@ -466,9 +465,7 @@ class SeqEncStep(nn.Module):
         assert isinstance(
             out,
             tuple,
-        ), (
-            f"out is not a tuple: {out}, type: {type(out)}, class: {self.__class__.__name__}"
-        )
+        ), f"out is not a tuple: {out}, type: {type(out)}, class: {self.__class__.__name__}"
         assert len(out) == len(self.out_keys)
         state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
         return state
@@ -501,7 +498,7 @@ class LinearInputEncoderStep(SeqEncStep):
         self.layer = nn.Linear(num_features, emsize, bias=bias)
         self.replace_nan_by_zero = replace_nan_by_zero
 
-    def _fit(self, *x: torch.Tensor, **kwargs: Any) -> None:
+    def _fit(self, *x: torch.Tensor, **kwargs: Any):
         """Fit the encoder step. Does nothing for LinearInputEncoderStep."""
 
     def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
@@ -525,90 +522,6 @@ class LinearInputEncoderStep(SeqEncStep):
         return (self.layer(x),)
 
 
-class MLPInputEncoderStep(SeqEncStep):
-    """An MLP-based input encoder step."""
-
-    def __init__(
-        self,
-        *,
-        num_features: int,
-        emsize: int,
-        hidden_dim: int | None = None,
-        activation: str = "gelu",
-        num_layers: int = 2,
-        replace_nan_by_zero: bool = False,
-        bias: bool = True,
-        in_keys: tuple[str, ...] = ("main",),
-        out_keys: tuple[str, ...] = ("output",),
-    ):
-        """Initialize the MLPInputEncoderStep.
-
-        Args:
-            num_features: The number of input features.
-            emsize: The embedding size, i.e. the number of output features.
-            hidden_dim: The hidden dimension of the MLP. If None, defaults to emsize.
-            activation: The activation function to use. Either "gelu" or "relu".
-            num_layers: The number of layers in the MLP (minimum 2).
-            replace_nan_by_zero: Whether to replace NaN values in the input by zero. Defaults to False.
-            bias: Whether to use a bias term in the linear layers. Defaults to True.
-            in_keys: The keys of the input tensors. Defaults to ("main",).
-            out_keys: The keys to assign the output tensors to. Defaults to ("output",).
-        """
-        super().__init__(in_keys, out_keys)
-
-        if hidden_dim is None:
-            hidden_dim = emsize
-
-        if num_layers < 2:
-            raise ValueError("num_layers must be at least 2 for an MLP encoder")
-
-        self.replace_nan_by_zero = replace_nan_by_zero
-
-        if activation == "gelu":
-            act_fn = nn.GELU()
-        elif activation == "relu":
-            act_fn = nn.ReLU()
-        else:
-            raise ValueError(f"Unknown activation: {activation}")
-
-        layers = []
-        # First layer: input -> hidden
-        layers.append(nn.Linear(num_features, hidden_dim, bias=bias))
-        layers.append(act_fn)
-
-        # Hidden layers
-        for _ in range(num_layers - 2):
-            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
-            layers.append(act_fn)
-
-        # Output layer: hidden -> emsize
-        layers.append(nn.Linear(hidden_dim, emsize, bias=bias))
-
-        self.mlp = nn.Sequential(*layers)
-
-    def _fit(self, *x: torch.Tensor, **kwargs: Any) -> None:
-        """Fit the encoder step. Does nothing for MLPInputEncoderStep."""
-
-    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
-        """Apply the MLP transformation to the input.
-
-        Args:
-            *x: The input tensors to concatenate and transform.
-            **kwargs: Unused keyword arguments.
-
-        Returns:
-            A tuple containing the transformed tensor.
-        """
-        x = torch.cat(x, dim=-1)
-        if self.replace_nan_by_zero:
-            x = torch.nan_to_num(x, nan=0.0)  # type: ignore
-
-        # Ensure input tensor dtype matches the first layer's weight dtype
-        x = x.to(self.mlp[0].weight.dtype)
-
-        return (self.mlp(x),)
-
-
 class NanHandlingEncoderStep(SeqEncStep):
     """Encoder step to handle NaN and infinite values in the input."""
 
@@ -652,7 +565,7 @@ class NanHandlingEncoderStep(SeqEncStep):
         self,
         x: torch.Tensor,
         **kwargs: Any,
-    ) -> tuple[torch.Tensor, torch.Tensor | None]:
+    ) -> tuple[torch.Tensor, torch.Tensor]:
         """Replace NaN and infinite values in the input tensor.
 
         Args:
@@ -681,7 +594,11 @@ class NanHandlingEncoderStep(SeqEncStep):
 
 
 class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
-    """Encoder step to remove empty (constant) features."""
+    """Encoder step to remove empty (constant) features.
+    Was changed to NOT DO ANYTHING, the removal of empty features now
+    done elsewhere, but the saved model still needs this encoder step.
+    TODO: REMOVE.
+    """
 
     def __init__(self, **kwargs: Any):
         """Initialize the RemoveEmptyFeaturesEncoderStep.
@@ -690,16 +607,16 @@ class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
             **kwargs: Keyword arguments passed to the parent SeqEncStep.
         """
         super().__init__(**kwargs)
-        self.column_selection_mask = None
+        self.sel = None
 
     def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:
-        """Compute the non-empty feature selection mask on the training set.
+        """Compute the feature selection mask on the training set.
 
         Args:
             x: The input tensor.
             **kwargs: Additional keyword arguments (unused).
         """
-        self.column_selection_mask = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)
+        self.sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)
 
     def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
         """Remove empty features from the input tensor.
@@ -711,7 +628,7 @@ class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
         Returns:
             A tuple containing the transformed tensor with empty features removed.
         """
-        return (select_features(x, self.column_selection_mask),)
+        return (select_features(x, self.sel),)
 
 
 class RemoveDuplicateFeaturesEncoderStep(SeqEncStep):
@@ -948,9 +865,9 @@ class InputNormalizationEncoderStep(SeqEncStep):
             x = to_ranking_low_mem(x)
 
         if self.remove_outliers:
-            assert self.remove_outliers_sigma > 1.0, (
-                "remove_outliers_sigma must be > 1.0"
-            )
+            assert (
+                self.remove_outliers_sigma > 1.0
+            ), "remove_outliers_sigma must be > 1.0"
 
             x, _ = remove_outliers(
                 x,
@@ -1011,7 +928,7 @@ class FrequencyFeatureEncoderStep(SeqEncStep):
         x: torch.Tensor,
         single_eval_pos: int | None = None,
         categorical_inds: list[int] | None = None,
-    ) -> None:
+    ):
         """Fit the encoder step. Does nothing for FrequencyFeatureEncoderStep."""
 
     def _transform(
@@ -1019,7 +936,7 @@ class FrequencyFeatureEncoderStep(SeqEncStep):
         x: torch.Tensor,
         single_eval_pos: int | None = None,
         categorical_inds: list[int] | None = None,
-    ) -> tuple[torch.Tensor]:
+    ):
         """Add frequency-based features to the input tensor.
 
         Args:
@@ -1045,9 +962,9 @@ class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
 
     def __init__(
         self,
-        num_features: int,
-        emsize: int,
-        base_encoder,  # noqa: ANN001
+        num_features,
+        emsize,
+        base_encoder,
         num_embs: int = 1_000,
         **kwargs: Any,
     ):
@@ -1059,17 +976,15 @@ class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
         self.embedding = nn.Embedding(num_embs, emsize)
         self.base_encoder = base_encoder
 
-    def _fit(
-        self, x: torch.Tensor, single_eval_pos: int, categorical_inds: list[int]
-    ) -> None:
+    def _fit(self, x, single_eval_pos: int, categorical_inds: list[int]):
         pass
 
     def _transform(
         self,
-        x: torch.Tensor,
+        x,
         single_eval_pos: int,
         categorical_inds: list[int],
-    ) -> tuple[torch.Tensor]:
+    ):
         if categorical_inds is None:
             is_categorical = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
         else:
@@ -1108,6 +1023,31 @@ class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
         return (embs,)
 
 
+class StyleEncoder(nn.Module):
+    def __init__(self, num_hyperparameters, em_size):
+        super().__init__()
+        self.em_size = em_size
+        self.embedding = nn.Linear(num_hyperparameters, self.em_size)
+
+    def forward(self, hyperparameters):  # B x num_hps
+        return self.embedding(hyperparameters)
+
+
+def get_linear_encoder_generator(in_keys):
+    def get_linear_encoder(num_features, emsize):
+        return SequentialEncoder(
+            LinearInputEncoderStep(
+                num_features,
+                emsize,
+                in_keys=in_keys,
+                out_keys=["output"],
+            ),
+            output_key="output",
+        )
+
+    return get_linear_encoder
+
+
 ##### TARGET ENCODERS #####
 
 
@@ -1116,27 +1056,23 @@ class MulticlassClassificationTargetEncoder(SeqEncStep):
         super().__init__(**kwargs)
         self.unique_ys_ = None
 
-    def _fit(self, y: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
+    def _fit(self, y: torch.Tensor, single_eval_pos: int, **kwargs: Any):
         assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
         self.unique_ys_ = [
             torch.unique(y[:single_eval_pos, b_i]) for b_i in range(y.shape[1])
         ]
 
     @staticmethod
-    def flatten_targets(
-        y: torch.Tensor, unique_ys: torch.Tensor | None = None
-    ) -> torch.Tensor:
+    def flatten_targets(y: torch.Tensor, unique_ys: torch.Tensor | None = None):
         if unique_ys is None:
             unique_ys = torch.unique(y)
         return (y.unsqueeze(-1) > unique_ys).sum(axis=-1)
 
-    def _transform(
-        self, y: torch.Tensor, single_eval_pos: int | None = None
-    ) -> tuple[torch.Tensor]:
+    def _transform(self, y: torch.Tensor, single_eval_pos: int | None = None):
         assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
-        assert not (y.isnan().any() and self.training), (
-            "NaNs are not allowed in the target at this point during training (set to model.eval() if not in training)"
-        )
+        assert not (
+            y.isnan().any() and self.training
+        ), "NaNs are not allowed in the target at this point during training (set to model.eval() if not in training)"
         y_new = y.clone()
         for B in range(y.shape[1]):
             y_new[:, B, :] = self.flatten_targets(y[:, B, :], self.unique_ys_[B])
diff --git a/src/tabpfn/architectures/base/layer.py b/src/tabpfn/architectures/base/layer.py
index ff9e3f4..7e715fb 100644
--- a/src/tabpfn/architectures/base/layer.py
+++ b/src/tabpfn/architectures/base/layer.py
@@ -277,18 +277,18 @@ class PerFeatureEncoderLayer(Module):
         Returns:
             The transformer state passed through the encoder layer.
         """
-        assert len(state.shape) == 4, (
-            "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
-        )
+        assert (
+            len(state.shape) == 4
+        ), "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
         save_peak_mem_factor = self.save_peak_mem_factor
         if cache_trainset_representation and not single_eval_pos:
             assert self.self_attn_between_items.has_cached_kv
             save_peak_mem_factor = None
 
         if att_src is not None:
-            assert not self.multiquery_item_attention_for_test_set, (
-                "Not implemented yet."
-            )
+            assert (
+                not self.multiquery_item_attention_for_test_set
+            ), "Not implemented yet."
             assert not cache_trainset_representation, "Not implemented yet."
             assert not single_eval_pos, (
                 "single_eval_pos should not be set, as the train representation"
diff --git a/src/tabpfn/architectures/base/memory.py b/src/tabpfn/architectures/base/memory.py
index b3c63ce..06e9f15 100644
--- a/src/tabpfn/architectures/base/memory.py
+++ b/src/tabpfn/architectures/base/memory.py
@@ -3,17 +3,16 @@
 from __future__ import annotations
 
 import os
-from collections.abc import Callable, Sequence
+import warnings
+from collections.abc import Callable
 from types import MethodType
-from typing import TYPE_CHECKING, Any, Literal, Union
+from typing import Any, Literal
 
+import numpy as np
 import torch
 
 from tabpfn.settings import settings
 
-if TYPE_CHECKING:
-    from tabpfn.architectures.interface import Architecture
-
 os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings.pytorch.pytorch_cuda_alloc_conf
 SAVE_PEAK_MEM_FACTOR = 8
 
@@ -29,6 +28,8 @@ NUM_SAMPLES_PLUS_FEATURES = 6.5
 CELLS_FACTOR = 0.25
 CELLS_SQUARED_FACTOR = 1.3e-7
 
+TO_BYTES_CONVERSION = {"b": 1, "mb": 1e6, "gb": 1e9}
+
 
 def support_save_peak_mem_factor(method: MethodType) -> Callable:
     """Can be applied to a method acting on a tensor 'x' whose first dimension is a
@@ -62,9 +63,9 @@ def support_save_peak_mem_factor(method: MethodType) -> Callable:
         **kwargs: Any,
     ) -> torch.Tensor:
         assert isinstance(self, torch.nn.Module)
-        assert save_peak_mem_factor is None or allow_inplace, (
-            "The parameter save_peak_mem_factor only supported with 'allow_inplace' set"
-        )
+        assert (
+            save_peak_mem_factor is None or allow_inplace
+        ), "The parameter save_peak_mem_factor only supported with 'allow_inplace' set."
         assert isinstance(x, torch.Tensor)
 
         tensor_inputs = list(tuple(self.parameters()) + tuple(args))
@@ -104,97 +105,348 @@ def support_save_peak_mem_factor(method: MethodType) -> Callable:
     return method_
 
 
-MemorySavingMode = Union[bool, Literal["auto"], float, int]
-
-
-def set_save_peak_memory(model: Architecture, *, enabled: bool) -> None:
-    """Set the peak memory factor to the default value, if enabled."""
-    if enabled:
-        model.reset_save_peak_mem_factor(SAVE_PEAK_MEM_FACTOR)
-    else:
-        model.reset_save_peak_mem_factor(None)
+class MemoryUsageEstimator:
+    SAVE_PEAK_MEM_FACTOR = 8
+
+    @classmethod
+    def convert_units(
+        cls,
+        value: float,
+        from_unit: Literal["b", "mb", "gb"],
+        to_unit: Literal["b", "mb", "gb"],
+    ) -> float:
+        """Convert a value from one unit to another."""
+        if from_unit not in TO_BYTES_CONVERSION:
+            raise ValueError(
+                f"Invalid unit {from_unit}. Must be one of 'b', 'mb', or 'gb'.",
+            )
+        if to_unit not in TO_BYTES_CONVERSION:
+            raise ValueError(
+                f"Invalid unit {to_unit}. Must be one of 'b', 'mb', or 'gb'.",
+            )
 
+        return (value * TO_BYTES_CONVERSION[from_unit]) / TO_BYTES_CONVERSION[to_unit]
+
+    @classmethod
+    def convert_bytes_to_unit(
+        cls,
+        value: float,
+        unit: Literal["b", "mb", "gb"],
+    ) -> float:
+        """Convenience method to convert bytes to a different unit.
+
+        Args:
+            value: The number of bytes.
+            unit: The unit to convert to.
+
+        Returns:
+            The number of bytes in the new unit.
+        """
+        return cls.convert_units(value, "b", unit)
+
+    @classmethod
+    def estimate_memory_of_one_batch(
+        cls,
+        X: torch.Tensor,
+        model: torch.nn.Module,
+        *,
+        cache_kv: bool,
+        dtype_byte_size: int,
+        unit: Literal["b", "mb", "gb"] = "gb",
+        n_train_samples: int | None = None,
+    ) -> float:
+        """Estimate the memory usage of a single batch.
+
+        The calculation is done based on the assumption that save_peak_mem_factor
+        is not used (since this estimation is used to determine whether to use it).
+
+        Args:
+            X: The input tensor.
+            model: The model to estimate the memory usage of.
+            cache_kv: Whether key and value tensors are cached.
+            dtype_byte_size: The size of the data type in bytes.
+            unit: The unit to convert the memory usage to.
+            n_train_samples: The number of training samples (only for cache_kv mode)
+
+        Returns:
+            The estimated memory usage of a single batch.
+        """
+        assert len(X.shape) in (2, 3), "X must be a 2D or 3D tensor"
+
+        if cache_kv:
+            assert isinstance(
+                n_train_samples,
+                int,
+            ), "n_train_samples must be provided when cache_kv is True"
+
+        if unit not in TO_BYTES_CONVERSION:
+            raise ValueError(f"Invalid unit {unit}. Must be one of 'b', 'mb', or 'gb'.")
+
+        embedding_size = model.ninp
+        features_per_group = model.features_per_group
+
+        n_layers = None
+        # Assumes the model has only encoder blocks
+        if (
+            hasattr(model, "transformer_encoder")
+            and model.transformer_encoder is not None
+        ):
+            n_layers = len(model.transformer_encoder.layers)
+
+        # Guarding against future changes in the transformer model
+        # Ideally, there should be an API exposed in the model to get the
+        # number of layers
+        if n_layers is None:
+            n_layers = 12
+            warnings.warn(
+                "Could not estimate number of encoder/decoder layers in the "
+                "transformer model, defaulting to 12.",
+                stacklevel=2,
+            )
 
-def should_save_peak_mem(
-    memory_saving_mode: MemorySavingMode,
-    X_train_shape: tuple[int, int],
-    X_test_shape: tuple[int, int],
-    devices: Sequence[torch.device],
-    dtype_byte_size: int,
-) -> bool:
-    """Uses heuristics to determine whether to save peak memory.
+        n_samples, n_features = X.shape[-2], X.shape[-1]
+        n_batches = X.shape[0] if len(X.shape) == 3 else 1
 
-    The aim is not only to avoid running out of memory for larger datasets, but also to
-    make inference faster. Enabling/disabling memory saving optimally can have a big
-    impact on fit+predict speed, sometimes greater than 2x.
+        n_feature_groups = int(np.ceil(n_features / features_per_group)) + 1
 
-    See details in https://github.com/PriorLabs/TabPFN/pull/605.
-    """
-    if isinstance(memory_saving_mode, bool):
-        return memory_saving_mode
+        model_mem = sum(p.numel() for p in model.parameters()) * dtype_byte_size
+        X_mem = n_samples * n_feature_groups * dtype_byte_size
+        activation_mem = (
+            n_samples
+            * n_feature_groups
+            * embedding_size
+            * n_layers
+            * dtype_byte_size
+            * n_batches
+        )
 
-    if all(device.type == "mps" for device in devices):
-        # - Memory saving usually seems to be faster even for small datasets on MPS
-        # - Running out of memory is quite bad because it locks up the whole MacOS UI
-        return True
+        total_mem_bytes = model_mem + X_mem + activation_mem
 
-    if all(device.type == "cpu" for device in devices):
-        return _should_save_peak_mem_cpu(X_train_shape, X_test_shape)
+        if cache_kv:
+            cached_mem = (
+                n_train_samples  # type: ignore
+                * n_feature_groups
+                * embedding_size
+                * 2  # key and value
+                * n_layers
+                * dtype_byte_size
+            )
+            total_mem_bytes += cached_mem
+
+        return cls.convert_bytes_to_unit(total_mem_bytes, unit)
+
+    @classmethod
+    def _get_mps_free_memory(cls) -> float:
+        """Get available free memory for MPS devices.
+
+        Tries to use `torch.mps.recommended_max_memory()` (available in
+        PyTorch >= 2.5.0). As a fallback for older versions, it uses the
+        `pyobjc-framework-Metal` library to query the Metal device API
+        directly.
+
+        Raises:
+            ImportError: If `pyobjc-framework-Metal` is required but not installed.
+            RuntimeError: If no MPS device can be found.
+
+        Returns:
+            The estimated free memory in bytes.
+        """
+        if hasattr(torch.mps, "recommended_max_memory"):
+            recommended = torch.mps.recommended_max_memory()
+            if recommended is not None:
+                allocated = torch.mps.current_allocated_memory()
+                return recommended - allocated
+
+        try:
+            # Fallback to using Metal API if torch.mps.recommended_max_memory is
+            # not available as it is only available in PyTorch 2.5.0 and later.
+            from Metal import MTLCreateSystemDefaultDevice
+        except ImportError as err:
+            raise ImportError(
+                "pyobjc-framework-Metal is required to access the Metal "
+                "APIs for determining available free memory for MPS devices. "
+                "Please install it via `pip install pyobjc-framework-Metal`."
+            ) from err
+
+        mtl_device = MTLCreateSystemDefaultDevice()
+        if mtl_device is None:
+            raise RuntimeError("No MPS device found.")
+
+        recommended = mtl_device.recommendedMaxWorkingSetSize()
+        allocated = (
+            torch.mps.current_allocated_memory()
+            if hasattr(torch.mps, "current_allocated_memory")
+            else 0
+        )
+        return recommended - allocated
+
+    @classmethod
+    def get_max_free_memory(
+        cls,
+        device: torch.device,
+        *,
+        unit: Literal["b", "mb", "gb"] = "gb",
+        default_gb_cpu_if_failed_to_calculate: float,
+    ) -> float:
+        """How much memory to use at most in GB, the memory usage will be calculated
+        based on an estimation of the systems free memory.
+
+        For CUDA will use the free memory of the GPU. For CPU will default to 32 GB.
+
+        Returns:
+        -------
+        The maximum memory usage in GB.
+        """
+        # TODO(Arjun): Make it accept a value for GPU specified by the user
+
+        # TODO: Get System Stats and adapt to free memory for default case
+
+        if device.type.startswith("cpu"):
+            try:
+                free_memory = (
+                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
+                )
+            except AttributeError:
+                from tabpfn.utils import get_total_memory_windows
+
+                if os.name == "nt":
+                    free_memory = get_total_memory_windows()
+                else:
+                    warnings.warn(
+                        "Could not get system memory, defaulting to"
+                        f" {default_gb_cpu_if_failed_to_calculate} GB",
+                        RuntimeWarning,
+                        stacklevel=2,
+                    )
+                    free_memory = cls.convert_units(
+                        default_gb_cpu_if_failed_to_calculate,
+                        "gb",
+                        "b",
+                    )
+
+            except ValueError:
+                warnings.warn(
+                    "Could not get system memory, defaulting to"
+                    f" {default_gb_cpu_if_failed_to_calculate} GB",
+                    RuntimeWarning,
+                    stacklevel=2,
+                )
+                free_memory = cls.convert_units(
+                    default_gb_cpu_if_failed_to_calculate,
+                    "gb",
+                    "b",
+                )
+
+        elif device.type.startswith("cuda"):
+            t = torch.cuda.get_device_properties(0).total_memory
+            torch.cuda.memory_reserved(0)
+            a = torch.cuda.memory_allocated(0)
+            free_memory = t - a  # free inside reserved
+        elif device.type.startswith("mps"):
+            free_memory = cls._get_mps_free_memory()
+        else:
+            raise ValueError(f"Unknown device {device}")
+
+        return cls.convert_bytes_to_unit(free_memory, unit)
+
+    @classmethod
+    def estimate_memory_remainder_after_batch(
+        cls,
+        X: torch.Tensor,
+        model: torch.nn.Module,
+        *,
+        cache_kv: bool,
+        device: torch.device,
+        dtype_byte_size: int,
+        safety_factor: float,
+        n_train_samples: int | None = None,
+        max_free_mem: float | int | None = None,
+    ) -> float:
+        """Whether to save peak memory or not.
+
+        Args:
+            X: The input tensor.
+            model: The model to estimate the memory usage of.
+            cache_kv: Whether key and value tensors are cached.
+            device: The device to use.
+            dtype_byte_size: The size of the data type in bytes.
+            safety_factor: The safety factor to apply.
+            n_train_samples: The number of training samples (only for cache_kv mode)
+            max_free_mem: The amount of free memory available.
+
+        Returns:
+            The amount of free memory available after a batch is computed.
+        """
+        if max_free_mem is None:
+            max_free_mem = cls.get_max_free_memory(
+                device,
+                unit="gb",
+                default_gb_cpu_if_failed_to_calculate=DEFAULT_CPU_MEMORY_GB_IF_NOT_CUDA,
+            )
 
-    if all(device.type == "cuda" for device in devices):
-        return _should_save_peak_mem_cuda(
-            X_train_shape, X_test_shape, devices, dtype_byte_size
+        mem_per_batch = cls.estimate_memory_of_one_batch(
+            X,
+            model,
+            cache_kv=cache_kv,
+            dtype_byte_size=dtype_byte_size,
+            unit="gb",
+            n_train_samples=n_train_samples,
         )
 
-    # For an unrecognised device, enable memory saving to be safe.
-    return True
-
-
-def _should_save_peak_mem_cpu(
-    X_train_shape: tuple[int, int], X_test_shape: tuple[int, int]
-) -> bool:
-    # TODO: Refine the CPU heuristic.
-    return _get_num_cells(X_train_shape, X_test_shape) > 200_000
-
-
-def _should_save_peak_mem_cuda(
-    X_train_shape: tuple[int, int],
-    X_test_shape: tuple[int, int],
-    devices: Sequence[torch.device],
-    dtype_byte_size: int,
-) -> bool:
-    free_memory_bytes = min(_get_free_cuda_memory_bytes(device) for device in devices)
-
-    # Our baseline is 2 byte floats on an 80GB H100.
-    # We observe that the threshold shifts roughly linearly with GPU memory size, so we
-    # make that adjustment.
-    baseline_cell_threshold = 6_000_000
-    baseline_dtype_byte_size = 2
-    baseline_gpu_memory_bytes = 80e9
-    cell_threshold = baseline_cell_threshold * (
-        baseline_dtype_byte_size / dtype_byte_size
-    )
-    cell_threshold = cell_threshold * (free_memory_bytes / baseline_gpu_memory_bytes)
-
-    # If we have multiple GPUs, we reduce the threshold a bit, based on empirical
-    # results.
-    if len(devices) > 1:
-        cell_threshold *= 0.8
-
-    return _get_num_cells(X_train_shape, X_test_shape) > cell_threshold
-
-
-def _get_free_cuda_memory_bytes(device: torch.device) -> float:
-    system_free_memory, _ = torch.cuda.mem_get_info(device)
-    pytorch_cache_free_memory = torch.cuda.memory_reserved(
-        device
-    ) - torch.cuda.memory_allocated(device)
-    return system_free_memory + pytorch_cache_free_memory
-
-
-def _get_num_cells(
-    X_train_shape: tuple[int, int], X_test_shape: tuple[int, int]
-) -> int:
-    n_train, n_features = X_train_shape
-    n_test, _ = X_test_shape
-    return (n_train + n_test) * n_features
+        return max_free_mem - (mem_per_batch * safety_factor)
+
+    @classmethod
+    def reset_peak_memory_if_required(
+        cls,
+        save_peak_mem: bool | Literal["auto"] | float | int,
+        model: torch.nn.Module,
+        X: torch.Tensor,
+        *,
+        cache_kv: bool,
+        device: torch.device,
+        dtype_byte_size: int,
+        safety_factor: float = 5.0,
+        n_train_samples: int | None = None,
+    ) -> None:
+        """Reset the peak memory if required.
+
+        Args:
+            save_peak_mem (bool | "auto" | float | int): If bool, specifies whether to
+                save peak memory or not.
+                If "auto", the amount of free memory is estimated and the option is
+                enabled or disabled based on the estimated usage.
+                If float or int, it is considered as the amount of memory available
+                (in GB) explicitly specified by the user. In this case, this value is
+                used to estimate whether or not to save peak memory.
+            model (torch.nn.Module): The model to reset the peak memory of.
+            X (torch.Tensor): The input tensor.
+            cache_kv (bool): Whether key and value tensors are cached.
+            device (torch.device): The device to use.
+            dtype_byte_size (int): The size of the data type in bytes.
+            safety_factor (float): The safety factor to apply.
+            n_train_samples (int): The number of training samples (to be used
+                only for cache_kv mode)
+        """
+        save_peak_mem_is_num = isinstance(
+            save_peak_mem,
+            (float, int),
+        ) and not isinstance(save_peak_mem, bool)
+        if save_peak_mem == "auto" or save_peak_mem_is_num:
+            memory_available_after_batch = cls.estimate_memory_remainder_after_batch(
+                X,
+                model,
+                cache_kv=cache_kv,
+                device=device,
+                dtype_byte_size=dtype_byte_size,
+                safety_factor=safety_factor,
+                n_train_samples=n_train_samples,
+                max_free_mem=save_peak_mem
+                if isinstance(save_peak_mem, (float, int))
+                else None,
+            )
+            save_peak_mem = memory_available_after_batch < 0
+
+        if save_peak_mem:
+            model.reset_save_peak_mem_factor(cls.SAVE_PEAK_MEM_FACTOR)
+        else:
+            model.reset_save_peak_mem_factor(None)
diff --git a/src/tabpfn/architectures/base/preprocessing.py b/src/tabpfn/architectures/base/preprocessing.py
new file mode 100644
index 0000000..04442b5
--- /dev/null
+++ b/src/tabpfn/architectures/base/preprocessing.py
@@ -0,0 +1,1487 @@
+#  Copyright (c) Prior Labs GmbH 2025.
+
+from __future__ import annotations
+
+import contextlib
+import hashlib
+from abc import abstractmethod
+from collections import UserList
+from collections.abc import Sequence
+from copy import deepcopy
+from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar
+from typing_extensions import Self, override
+
+import numpy as np
+import torch
+from scipy import optimize
+from scipy.stats import shapiro
+from sklearn.compose import ColumnTransformer, make_column_selector
+from sklearn.decomposition import TruncatedSVD
+from sklearn.impute import SimpleImputer
+from sklearn.pipeline import FeatureUnion, Pipeline
+from sklearn.preprocessing import (
+    FunctionTransformer,
+    MinMaxScaler,
+    OneHotEncoder,
+    OrdinalEncoder,
+    PowerTransformer,
+    QuantileTransformer,
+    RobustScaler,
+    StandardScaler,
+)
+
+from tabpfn.utils import infer_random_state
+
+if TYPE_CHECKING:
+    from sklearn.base import TransformerMixin
+
+
+try:
+    from kditransform import KDITransformer
+
+    # This import fails on some systems, due to problems with numba
+except ImportError:
+    KDITransformer = PowerTransformer  # fallback to avoid error
+
+
+class KDITransformerWithNaN(KDITransformer):
+    """KDI transformer that can handle NaN values. It performs KDI with NaNs replaced by
+    mean values and then fills the NaN values with NaNs after the transformation.
+    """
+
+    def _more_tags(self) -> dict:
+        return {"allow_nan": True}
+
+    def fit(self, X: torch.Tensor | np.ndarray, y: Any | None = None) -> Self:
+        if isinstance(X, torch.Tensor):
+            X = X.cpu().numpy()
+
+        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
+        return super().fit(X, y)  # type: ignore
+
+    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
+        # if tensor convert to numpy
+        if isinstance(X, torch.Tensor):
+            X = X.cpu().numpy()
+
+        # Calculate the NaN mask for the current dataset
+        nan_mask = np.isnan(X)
+
+        # Replace NaNs with the mean of columns
+        imputation = np.nanmean(X, axis=0)
+        imputation = np.nan_to_num(imputation, nan=0)
+        X = np.nan_to_num(X, nan=imputation)
+
+        # Apply the transformation
+        X = super().transform(X)
+
+        # Reintroduce NaN values based on the current dataset's mask
+        X[nan_mask] = np.nan
+
+        return X  # type: ignore
+
+
+class AdaptiveQuantileTransformer(QuantileTransformer):
+    """A QuantileTransformer that automatically adapts the 'n_quantiles' parameter
+    based on the number of samples provided during the 'fit' method.
+
+    This fixes an issue in older versions of scikit-learn where the 'n_quantiles'
+    parameter could not exceed the number of samples in the input data.
+
+    This code prevents errors that occur when the requested 'n_quantiles' is
+    greater than the number of available samples in the input data (X).
+    This situation can arises because we first initialize the transformer
+    based on total samples and then subsample.
+    """
+
+    def __init__(self, *, n_quantiles: int = 1000, **kwargs: Any) -> None:
+        # Store the user's desired n_quantiles to use as an upper bound
+        self._user_n_quantiles = n_quantiles
+        # Initialize parent with this, but it will be adapted in fit
+        super().__init__(n_quantiles=n_quantiles, **kwargs)
+
+    def fit(
+        self, X: np.ndarray, y: np.ndarray | None = None
+    ) -> AdaptiveQuantileTransformer:
+        n_samples = X.shape[0]
+
+        # Adapt n_quantiles for this fit: min of user's preference and available samples
+        # Ensure n_quantiles is at least 1
+        effective_n_quantiles = max(
+            1, min(self._user_n_quantiles, n_samples, self.subsample)
+        )
+
+        # Set self.n_quantiles to the effective value BEFORE calling super().fit()
+        # This ensures the parent class uses the adapted value for fitting
+        # and self.n_quantiles will reflect the value used for the fit.
+        self.n_quantiles = effective_n_quantiles
+
+        # Convert Generator to RandomState if needed for sklearn compatibility
+        if isinstance(self.random_state, np.random.Generator):
+            # Generate a random integer to use as seed for RandomState
+            seed = int(self.random_state.integers(0, 2**32))
+            self.random_state = np.random.RandomState(seed)
+        elif hasattr(self.random_state, "bit_generator"):
+            # Handle other Generator-like objects
+            raise ValueError(
+                f"Unsupported random state type: {type(self.random_state)}. "
+                "Please provide an integer seed or np.random.RandomState object."
+            )
+
+        return super().fit(X, y)
+
+
+ALPHAS = (
+    0.05,
+    0.1,
+    0.2,
+    0.25,
+    0.3,
+    0.4,
+    0.5,
+    0.6,
+    0.8,
+    1.0,
+    1.2,
+    1.5,
+    1.8,
+    2.0,
+    2.5,
+    3.0,
+    5.0,
+)
+
+
+def get_all_kdi_transformers() -> dict[str, KDITransformerWithNaN]:
+    try:
+        all_preprocessors = {
+            "kdi": KDITransformerWithNaN(alpha=1.0, output_distribution="normal"),
+            "kdi_uni": KDITransformerWithNaN(
+                alpha=1.0,
+                output_distribution="uniform",
+            ),
+        }
+        for alpha in ALPHAS:
+            all_preprocessors[f"kdi_alpha_{alpha}"] = KDITransformerWithNaN(
+                alpha=alpha,
+                output_distribution="normal",
+            )
+            all_preprocessors[f"kdi_alpha_{alpha}_uni"] = KDITransformerWithNaN(
+                alpha=alpha,
+                output_distribution="uniform",
+            )
+        return all_preprocessors
+    except Exception:  # noqa: BLE001
+        return {}
+
+
+# this is taken from https://github.com/scipy/scipy/pull/18852
+# which fix overflow issues
+# we can directly import from scipy once we drop support for scipy < 1.12
+def _yeojohnson(x, lmbda=None):
+    x = np.asarray(x)
+    if x.size == 0:
+        # changed from scipy from return x
+        return (x, None) if lmbda is None else x
+
+    if np.issubdtype(x.dtype, np.complexfloating):
+        raise ValueError(
+            "Yeo-Johnson transformation is not defined for " "complex numbers."
+        )
+
+    if np.issubdtype(x.dtype, np.integer):
+        x = x.astype(np.float64, copy=False)
+
+    if lmbda is not None:
+        return _yeojohnson_transform(x, lmbda)
+
+    # if lmbda=None, find the lmbda that maximizes the log-likelihood function.
+    lmax = _yeojohnson_normmax(x)
+    y = _yeojohnson_transform(x, lmax)
+
+    return y, lmax
+
+
+def _yeojohnson_transform(x, lmbda):
+    """Returns `x` transformed by the Yeo-Johnson power transform with given
+    parameter `lmbda`.
+    """
+    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
+    out = np.zeros_like(x, dtype=dtype)
+    pos = x >= 0  # binary mask
+
+    # when x >= 0
+    if abs(lmbda) < np.spacing(1.0):
+        out[pos] = np.log1p(x[pos])
+    else:  # lmbda != 0
+        # more stable version of: ((x + 1) ** lmbda - 1) / lmbda
+        out[pos] = np.expm1(lmbda * np.log1p(x[pos])) / lmbda
+
+    # when x < 0
+    if abs(lmbda - 2) > np.spacing(1.0):
+        out[~pos] = -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda)
+    else:  # lmbda == 2
+        out[~pos] = -np.log1p(-x[~pos])
+
+    return out
+
+
+def _yeojohnson_llf(lmb, data):
+    r"""The yeojohnson log-likelihood function."""
+    data = np.asarray(data)
+    n_samples = data.shape[0]
+
+    if n_samples == 0:
+        return np.nan
+
+    trans = _yeojohnson_transform(data, lmb)
+    trans_var = trans.var(axis=0)
+    loglike = np.empty_like(trans_var)
+
+    # Avoid RuntimeWarning raised by np.log when the variance is too low
+    tiny_variance = trans_var < np.finfo(trans_var.dtype).tiny
+    loglike[tiny_variance] = np.inf
+
+    loglike[~tiny_variance] = -n_samples / 2 * np.log(trans_var[~tiny_variance])
+    loglike[~tiny_variance] += (lmb - 1) * (np.sign(data) * np.log1p(np.abs(data))).sum(
+        axis=0
+    )
+    return loglike
+
+
+def _yeojohnson_normmax(x, brack=None):
+    """Compute optimal Yeo-Johnson transform parameter.
+    Compute optimal Yeo-Johnson transform parameter for input data, using
+    maximum likelihood estimation.
+
+    """
+
+    def _neg_llf(lmbda, data):
+        llf = _yeojohnson_llf(lmbda, data)
+        # reject likelihoods that are inf which are likely due to small
+        # variance in the transformed space
+        llf[np.isinf(llf)] = -np.inf
+        return -llf
+
+    with np.errstate(invalid="ignore"):
+        if not np.all(np.isfinite(x)):
+            raise ValueError("Yeo-Johnson input must be finite.")
+        if np.all(x == 0):
+            return 1.0
+        if brack is not None:
+            return optimize.brent(_neg_llf, brack=brack, args=(x,))
+        x = np.asarray(x)
+        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
+        # Allow values up to 20 times the maximum observed value to be safely
+        # transformed without over- or underflow.
+        log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
+        # Use half of floating point's exponent range to allow safe computation
+        # of the variance of the transformed data.
+        log_eps = np.log(np.finfo(dtype).eps)
+        log_tiny_float = (np.log(np.finfo(dtype).tiny) - log_eps) / 2
+        log_max_float = (np.log(np.finfo(dtype).max) + log_eps) / 2
+        # Compute the bounds by approximating the inverse of the Yeo-Johnson
+        # transform on the smallest and largest floating point exponents, given
+        # the largest data we expect to observe. See [1] for further details.
+        # [1] https://github.com/scipy/scipy/pull/18852#issuecomment-1630286174
+        lb = log_tiny_float / log1p_max_x
+        ub = log_max_float / log1p_max_x
+        # Convert the bounds if all or some of the data is negative.
+        if np.all(x < 0):
+            lb, ub = 2 - ub, 2 - lb
+        elif np.any(x < 0):
+            lb, ub = max(2 - ub, lb), min(2 - lb, ub)
+        # Match `optimize.brent`'s tolerance.
+        tol_brent = 1.48e-08
+        return optimize.fminbound(_neg_llf, lb, ub, args=(x,), xtol=tol_brent)
+
+
+# we created this inspired by the scipy change for transform
+# https://github.com/scipy/scipy/pull/18852
+# this is not in scipy even 1.12
+def _yeojohnson_inverse_transform(x, lmbda):
+    """Return inverse-transformed input x following Yeo-Johnson inverse
+    transform with parameter lambda.
+    """
+    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
+    x_inv = np.zeros_like(x, dtype=dtype)
+    pos = x >= 0
+
+    # when x >= 0
+    if abs(lmbda) < np.spacing(1.0):
+        x_inv[pos] = np.expm1(x[pos])
+    else:  # lmbda != 0
+        # more stable version of: (x * lmbda + 1) ** (1 / lmbda) - 1
+        x_inv[pos] = np.expm1(np.log(x[pos] * lmbda + 1) / lmbda)
+
+    # when x < 0
+    if abs(lmbda - 2) > np.spacing(1.0):
+        # more stable version of: 1 - (-(2 - lmbda) * x + 1) ** (1 / (2 - lmbda))
+        x_inv[~pos] = -np.expm1(np.log(-(2 - lmbda) * x[~pos] + 1) / (2 - lmbda))
+    else:  # lmbda == 2
+        x_inv[~pos] = -np.expm1(-x[~pos])
+
+    return x_inv
+
+
+class SafePowerTransformer(PowerTransformer):
+    """Variant of PowerTransformer that uses the scipy yeo-johnson functions
+    which have been fixed to avoid overflow issues.
+    """
+
+    def __init__(
+        self,
+        method: str = "yeo-johnson",
+        *,
+        standardize: bool = True,
+        copy: bool = True,
+    ) -> None:
+        super().__init__(method=method, standardize=standardize, copy=copy)
+
+    # requires scipy >= 1.9
+    # this is the default in scikit-learn main for versions > 1.7
+    # see https://github.com/scikit-learn/scikit-learn/pull/31227
+    def _yeo_johnson_optimize(self, x):
+        # the computation of lambda is influenced by NaNs so we need to
+        # get rid of them
+        x = x[~np.isnan(x)]
+        _, lmbda = _yeojohnson(x, lmbda=None)
+        return lmbda
+
+    def _yeo_johnson_transform(self, x, lmbda):
+        return _yeojohnson_transform(x, lmbda)
+
+    def _yeo_johnson_inverse_transform(self, x, lmbda):
+        return _yeojohnson_inverse_transform(x, lmbda)
+
+
+def skew(x: np.ndarray) -> float:
+    """skewness: 3 * (mean - median) / std."""
+    return float(3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0))
+
+
+def _inf_to_nan_func(x: np.ndarray) -> np.ndarray:
+    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)
+
+
+def _exp_minus_1(x: np.ndarray) -> np.ndarray:
+    return np.exp(x) - 1  # type: ignore
+
+
+T = TypeVar("T")
+
+
+def _identity(x: T) -> T:
+    return x
+
+
+inf_to_nan_transformer = FunctionTransformer(
+    func=_inf_to_nan_func,
+    inverse_func=_identity,
+    check_inverse=False,
+)
+nan_impute_transformer = SimpleImputer(
+    missing_values=np.nan,
+    strategy="mean",
+    # keep empty features for inverse to function
+    keep_empty_features=True,
+)
+nan_impute_transformer.inverse_transform = (
+    _identity  # do not inverse np.nan values.  # type: ignore
+)
+
+_make_finite_transformer = [
+    ("inf_to_nan", inf_to_nan_transformer),
+    ("nan_impute", nan_impute_transformer),
+]
+
+
+def make_standard_scaler_safe(
+    _name_scaler_tuple: tuple[str, TransformerMixin],
+    *,
+    no_name: bool = False,
+) -> Pipeline:
+    # Make sure that all data that enters and leaves a scaler is finite.
+    # This is needed in edge cases where, for example, a division by zero
+    # occurs while scaling or when the input contains not number values.
+    return Pipeline(
+        steps=[
+            *[(n + "_pre ", deepcopy(t)) for n, t in _make_finite_transformer],
+            ("placeholder", _name_scaler_tuple) if no_name else _name_scaler_tuple,
+            *[(n + "_post", deepcopy(t)) for n, t in _make_finite_transformer],
+        ],
+    )
+
+
+def make_box_cox_safe(input_transformer: TransformerMixin | Pipeline) -> Pipeline:
+    """Make box cox save.
+
+    The Box-Cox transformation can only be applied to strictly positive data.
+    With first MinMax scaling, we achieve this without loss of function.
+    Additionally, for test data, we also need clipping.
+    """
+    return Pipeline(
+        steps=[
+            ("mm", MinMaxScaler(feature_range=(0.1, 1), clip=True)),
+            ("box_cox", input_transformer),
+        ],
+    )
+
+
+def add_safe_standard_to_safe_power_without_standard(
+    input_transformer: TransformerMixin,
+) -> Pipeline:
+    """In edge cases PowerTransformer can create inf values and similar. Then, the post
+    standard scale crashes. This fixes this issue.
+    """
+    return Pipeline(
+        steps=[
+            ("input_transformer", input_transformer),
+            ("standard", make_standard_scaler_safe(("standard", StandardScaler()))),
+        ],
+    )
+
+
+class _TransformResult(NamedTuple):
+    X: np.ndarray
+    categorical_features: list[int]
+
+
+# TODO(eddiebergman): I'm sure there's a way to handle this when using dataframes.
+class FeaturePreprocessingTransformerStep:
+    """Base class for feature preprocessing steps.
+
+    It's main abstraction is really just to provide categorical indices along the
+    pipeline.
+    """
+
+    categorical_features_after_transform_: list[int]
+
+    def fit_transform(
+        self,
+        X: np.ndarray,
+        categorical_features: list[int],
+    ) -> _TransformResult:
+        self.fit(X, categorical_features)
+        # TODO(eddiebergman): If we could get rid of this... anywho, needed for
+        # the AddFingerPrint
+        result = self._transform(X, is_test=False)
+        return _TransformResult(result, self.categorical_features_after_transform_)
+
+    @abstractmethod
+    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
+        """Underlying method of the preprocessor to implement by subclassses.
+
+        Args:
+            X: 2d array of shape (n_samples, n_features)
+            categorical_features: list of indices of categorical feature.
+
+        Returns:
+            list of indices of categorical features after the transform.
+        """
+        raise NotImplementedError
+
+    def fit(self, X: np.ndarray, categorical_features: list[int]) -> Self:
+        """Fits the preprocessor.
+
+        Args:
+            X: 2d array of shape (n_samples, n_features)
+            categorical_features: list of indices of categorical feature.
+        """
+        self.categorical_features_after_transform_ = self._fit(X, categorical_features)
+        assert self.categorical_features_after_transform_ is not None, (
+            "_fit should have returned a list of the indexes of the categorical"
+            "features after the transform."
+        )
+        return self
+
+    @abstractmethod
+    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
+        """Underlying method of the preprocessor to implement by subclassses.
+
+        Args:
+            X: 2d array of shape (n_samples, n_features)
+            is_test: Should be removed, used for the `AddFingerPrint` step.
+
+        Returns:
+            2d np.ndarray of shape (n_samples, new n_features)
+        """
+        raise NotImplementedError
+
+    def transform(self, X: np.ndarray) -> _TransformResult:
+        """Transforms the data.
+
+        Args:
+            X: 2d array of shape (n_samples, n_features).
+        """
+        # TODO: Get rid of this, it's always test in `transform`
+        result = self._transform(X, is_test=True)
+        return _TransformResult(result, self.categorical_features_after_transform_)
+
+
+class SequentialFeatureTransformer(UserList):
+    """A transformer that applies a sequence of feature preprocessing steps.
+    This is very related to sklearn's Pipeline, but it is designed to work with
+    categorical_features lists that are always passed on.
+
+    Currently this class is only used once, thus this could also be made
+    less general if needed.
+    """
+
+    def __init__(self, steps: Sequence[FeaturePreprocessingTransformerStep]):
+        super().__init__(steps)
+        self.steps = steps
+        self.categorical_features_: list[int] | None = None
+
+    def fit_transform(
+        self,
+        X: np.ndarray | torch.tensor,
+        categorical_features: list[int],
+    ) -> _TransformResult:
+        """Fit and transform the data using the fitted pipeline.
+
+        Args:
+            X: 2d array of shape (n_samples, n_features)
+            categorical_features: list of indices of categorical features.
+        """
+        for step in self.steps:
+            X, categorical_features = step.fit_transform(X, categorical_features)
+            assert isinstance(categorical_features, list), (
+                f"The {step=} must return list of categorical features,"
+                f" but {type(step)} returned {categorical_features}"
+            )
+
+        self.categorical_features_ = categorical_features
+        return _TransformResult(X, categorical_features)
+
+    def fit(
+        self, X: np.ndarray | torch.tensor, categorical_features: list[int]
+    ) -> Self:
+        """Fit all the steps in the pipeline.
+
+        Args:
+            X: 2d array of shape (n_samples, n_features)
+            categorical_features: list of indices of categorical feature.
+        """
+        assert (
+            len(self) > 0
+        ), "The SequentialFeatureTransformer must have at least one step."
+        self.fit_transform(X, categorical_features)
+        return self
+
+    def transform(self, X: np.ndarray) -> _TransformResult:
+        """Transform the data using the fitted pipeline.
+
+        Args:
+            X: 2d array of shape (n_samples, n_features).
+        """
+        assert (
+            len(self) > 0
+        ), "The SequentialFeatureTransformer must have at least one step."
+        assert self.categorical_features_ is not None, (
+            "The SequentialFeatureTransformer must be fit before it"
+            " can be used to transform."
+        )
+        categorical_features = []
+        for step in self:
+            X, categorical_features = step.transform(X)
+
+        assert categorical_features == self.categorical_features_, (
+            f"Expected categorical features {self.categorical_features_},"
+            f"but got {categorical_features}"
+        )
+        return _TransformResult(X, categorical_features)
+
+
+class RemoveConstantFeaturesStep(FeaturePreprocessingTransformerStep):
+    """Remove features that are constant in the training data."""
+
+    def __init__(self) -> None:
+        super().__init__()
+        self.sel_: list[bool] | None = None
+
+    @override
+    def _fit(
+        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
+    ) -> list[int]:
+        if isinstance(X, torch.Tensor):
+            sel_ = torch.max(X[0:1, :] != X, dim=0)[0].cpu()
+        else:
+            sel_ = ((X[0:1, :] == X).mean(axis=0) < 1.0).tolist()
+
+        if not any(sel_):
+            raise ValueError(
+                "All features are constant and would have been removed!"
+                " Unable to predict using TabPFN.",
+            )
+        self.sel_ = sel_
+
+        return [
+            new_idx
+            for new_idx, idx in enumerate(np.where(sel_)[0])
+            if idx in categorical_features
+        ]
+
+    @override
+    def _transform(
+        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
+    ) -> np.ndarray:
+        assert self.sel_ is not None, "You must call fit first"
+        return X[:, self.sel_]
+
+
+_CONSTANT = 10**12
+
+
+def float_hash_arr(arr: np.ndarray) -> float:
+    _hash = int(hashlib.sha256(arr.tobytes()).hexdigest(), 16)
+    return _hash % _CONSTANT / _CONSTANT
+
+
+class AddFingerprintFeaturesStep(FeaturePreprocessingTransformerStep):
+    """Adds a fingerprint feature to the features based on hash of each row.
+
+    If `is_test = True`, it keeps the first hash even if there are collisions.
+    If `is_test = False`, it handles hash collisions by counting up and rehashing
+    until a unique hash is found.
+    The idea is basically to add a random feature to help the model distinguish between
+    identical rows. We use hashing to make sure the result does not depend on the order
+    of the rows.
+    """
+
+    def __init__(self, random_state: int | np.random.Generator | None = None):
+        super().__init__()
+        self.random_state = random_state
+
+    @override
+    def _fit(
+        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
+    ) -> list[int]:
+        _, rng = infer_random_state(self.random_state)
+        self.rnd_salt_ = int(rng.integers(0, 2**16))
+        return [*categorical_features]
+
+    @override
+    def _transform(
+        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
+    ) -> np.ndarray | torch.Tensor:
+        X_det = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
+
+        # no detach necessary for numpy
+        X_h = np.zeros(X.shape[0], dtype=X_det.dtype)
+        if is_test:
+            # Keep the first hash even if there are collisions
+            salted_X = X_det + self.rnd_salt_
+            for i, row in enumerate(salted_X):
+                h = float_hash_arr(row + self.rnd_salt_)
+                X_h[i] = h
+        else:
+            # Handle hash collisions by counting up and rehashing
+            seen_hashes = set()
+            salted_X = X_det + self.rnd_salt_
+            for i, row in enumerate(salted_X):
+                h = float_hash_arr(row)
+                add_to_hash = 0
+                while h in seen_hashes and not np.isnan(row).all():
+                    add_to_hash += 1
+                    h = float_hash_arr(row + add_to_hash)
+                X_h[i] = h
+                seen_hashes.add(h)
+
+        if isinstance(X, torch.Tensor):
+            return torch.cat(
+                [X, torch.from_numpy(X_h).float().reshape(-1, 1).to(X.device)], dim=1
+            )
+        else:  # noqa: RET505
+            return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)
+
+
+class ShuffleFeaturesStep(FeaturePreprocessingTransformerStep):
+    """Shuffle the features in the data."""
+
+    def __init__(
+        self,
+        shuffle_method: Literal["shuffle", "rotate"] | None = "rotate",
+        shuffle_index: int = 0,
+        random_state: int | np.random.Generator | None = None,
+    ):
+        super().__init__()
+        self.random_state = random_state
+        self.shuffle_method = shuffle_method
+        self.shuffle_index = shuffle_index
+
+        self.index_permutation_: list[int] | None = None
+
+    @override
+    def _fit(
+        self, X: np.ndarray | torch.tensor, categorical_features: list[int]
+    ) -> list[int]:
+        _, rng = infer_random_state(self.random_state)
+        if self.shuffle_method == "rotate":
+            index_permutation = np.roll(
+                np.arange(X.shape[1]),
+                self.shuffle_index,
+            ).tolist()
+        elif self.shuffle_method == "shuffle":
+            index_permutation = rng.permutation(X.shape[1]).tolist()
+        elif self.shuffle_method is None:
+            index_permutation = np.arange(X.shape[1]).tolist()
+        else:
+            raise ValueError(f"Unknown shuffle method {self.shuffle_method}")
+        if isinstance(X, torch.Tensor):
+            self.index_permutation_ = torch.tensor(index_permutation, dtype=torch.long)
+        else:
+            self.index_permutation_ = index_permutation
+
+        return [
+            new_idx
+            for new_idx, idx in enumerate(index_permutation)
+            if idx in categorical_features
+        ]
+
+    @override
+    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
+        assert self.index_permutation_ is not None, "You must call fit first"
+        assert (
+            len(self.index_permutation_) == X.shape[1]
+        ), "The number of features must not change after fit"
+        return X[:, self.index_permutation_]
+
+
+class ReshapeFeatureDistributionsStep(FeaturePreprocessingTransformerStep):
+    """Reshape the feature distributions using different transformations."""
+
+    APPEND_TO_ORIGINAL_THRESHOLD = 500
+
+    @staticmethod
+    def get_column_types(X: np.ndarray) -> list[str]:
+        """Returns a list of column types for the given data, that indicate how
+        the data should be preprocessed.
+        """
+        # TODO(eddiebergman): Bad to keep calling skew again and again here...
+        column_types = []
+        for col in range(X.shape[1]):
+            if np.unique(X[:, col]).size < 10:
+                column_types.append(f"ordinal_{col}")
+            elif (
+                skew(X[:, col]) > 1.1
+                and np.min(X[:, col]) >= 0
+                and np.max(X[:, col]) <= 1
+            ):
+                column_types.append(f"skewed_pos_1_0_{col}")
+            elif skew(X[:, col]) > 1.1 and np.min(X[:, col]) > 0:
+                column_types.append(f"skewed_pos_{col}")
+            elif skew(X[:, col]) > 1.1:
+                column_types.append(f"skewed_{col}")
+            elif shapiro(X[0:3000, col]).statistic > 0.95:
+                column_types.append(f"normal_{col}")
+            else:
+                column_types.append(f"other_{col}")
+        return column_types
+
+    @staticmethod
+    def get_adaptive_preprocessors(
+        num_examples: int = 100,
+        random_state: int | None = None,
+    ) -> dict[str, ColumnTransformer]:
+        """Returns a dictionary of adaptive column transformers that can be used to
+        preprocess the data. Adaptive column transformers are used to preprocess the
+        data based on the column type, they receive a pandas dataframe with column
+        names, that indicate the column type. Column types are not datatypes,
+        but rather a string that indicates how the data should be preprocessed.
+
+        Args:
+            num_examples: The number of examples in the dataset.
+            random_state: The random state to use for the transformers.
+        """
+        return {
+            "adaptive": ColumnTransformer(
+                [
+                    (
+                        "skewed_pos_1_0",
+                        FunctionTransformer(
+                            func=np.exp,
+                            inverse_func=np.log,
+                            check_inverse=False,
+                        ),
+                        make_column_selector("skewed_pos_1_0*"),
+                    ),
+                    (
+                        "skewed_pos",
+                        make_box_cox_safe(
+                            add_safe_standard_to_safe_power_without_standard(
+                                SafePowerTransformer(
+                                    standardize=False,
+                                    method="box-cox",
+                                ),
+                            ),
+                        ),
+                        make_column_selector("skewed_pos*"),
+                    ),
+                    (
+                        "skewed",
+                        add_safe_standard_to_safe_power_without_standard(
+                            SafePowerTransformer(
+                                standardize=False,
+                                method="yeo-johnson",
+                            ),
+                        ),
+                        make_column_selector("skewed*"),
+                    ),
+                    (
+                        "other",
+                        AdaptiveQuantileTransformer(
+                            output_distribution="normal",
+                            n_quantiles=max(num_examples // 10, 2),
+                            random_state=random_state,
+                        ),
+                        # "other" or "ordinal"
+                        make_column_selector("other*"),
+                    ),
+                    (
+                        "ordinal",
+                        # default FunctionTransformer yields the identity function
+                        FunctionTransformer(),
+                        # "other" or "ordinal"
+                        make_column_selector("ordinal*"),
+                    ),
+                    (
+                        "normal",
+                        # default FunctionTransformer yields the identity function
+                        FunctionTransformer(),
+                        make_column_selector("normal*"),
+                    ),
+                ],
+                remainder="passthrough",
+            ),
+        }
+
+    @staticmethod
+    def get_all_preprocessors(
+        num_examples: int,
+        random_state: int | None = None,
+    ) -> dict[str, TransformerMixin | Pipeline]:
+        all_preprocessors = {
+            "power": add_safe_standard_to_safe_power_without_standard(
+                PowerTransformer(standardize=False),
+            ),
+            "safepower": add_safe_standard_to_safe_power_without_standard(
+                SafePowerTransformer(standardize=False),
+            ),
+            "power_box": make_box_cox_safe(
+                add_safe_standard_to_safe_power_without_standard(
+                    PowerTransformer(standardize=False, method="box-cox"),
+                ),
+            ),
+            "safepower_box": make_box_cox_safe(
+                add_safe_standard_to_safe_power_without_standard(
+                    SafePowerTransformer(standardize=False, method="box-cox"),
+                ),
+            ),
+            "log": FunctionTransformer(
+                func=np.log,
+                inverse_func=np.exp,
+                check_inverse=False,
+            ),
+            "1_plus_log": FunctionTransformer(
+                func=np.log1p,
+                inverse_func=_exp_minus_1,
+                check_inverse=False,
+            ),
+            "exp": FunctionTransformer(
+                func=np.exp,
+                inverse_func=np.log,
+                check_inverse=False,
+            ),
+            "quantile_uni_coarse": AdaptiveQuantileTransformer(
+                output_distribution="uniform",
+                n_quantiles=max(num_examples // 10, 2),
+                random_state=random_state,
+            ),
+            "quantile_norm_coarse": AdaptiveQuantileTransformer(
+                output_distribution="normal",
+                n_quantiles=max(num_examples // 10, 2),
+                random_state=random_state,
+            ),
+            "quantile_uni": AdaptiveQuantileTransformer(
+                output_distribution="uniform",
+                n_quantiles=max(num_examples // 5, 2),
+                random_state=random_state,
+            ),
+            "quantile_norm": AdaptiveQuantileTransformer(
+                output_distribution="normal",
+                n_quantiles=max(num_examples // 5, 2),
+                random_state=random_state,
+            ),
+            "quantile_uni_fine": AdaptiveQuantileTransformer(
+                output_distribution="uniform",
+                n_quantiles=num_examples,
+                random_state=random_state,
+            ),
+            "quantile_norm_fine": AdaptiveQuantileTransformer(
+                output_distribution="normal",
+                n_quantiles=num_examples,
+                random_state=random_state,
+            ),
+            "robust": RobustScaler(unit_variance=True),
+            # default FunctionTransformer yields the identity function
+            "none": FunctionTransformer(),
+            **get_all_kdi_transformers(),
+        }
+
+        with contextlib.suppress(Exception):
+            all_preprocessors["norm_and_kdi"] = FeatureUnion(
+                [
+                    (
+                        "norm",
+                        AdaptiveQuantileTransformer(
+                            output_distribution="normal",
+                            n_quantiles=max(num_examples // 10, 2),
+                            random_state=random_state,
+                        ),
+                    ),
+                    (
+                        "kdi",
+                        KDITransformerWithNaN(alpha=1.0, output_distribution="uniform"),
+                    ),
+                ],
+            )
+
+        all_preprocessors.update(
+            ReshapeFeatureDistributionsStep.get_adaptive_preprocessors(
+                num_examples,
+                random_state=random_state,
+            ),
+        )
+
+        return all_preprocessors
+
+    def get_all_global_transformers(
+        self,
+        num_examples: int,
+        num_features: int,
+        random_state: int | None = None,
+    ) -> dict[str, FeatureUnion | Pipeline]:
+        return {
+            "scaler": make_standard_scaler_safe(("standard", StandardScaler())),
+            "svd": FeatureUnion(
+                [
+                    # default FunctionTransformer yields the identity function
+                    ("passthrough", FunctionTransformer()),
+                    (
+                        "svd",
+                        Pipeline(
+                            steps=[
+                                (
+                                    "save_standard",
+                                    make_standard_scaler_safe(
+                                        ("standard", StandardScaler(with_mean=False)),
+                                    ),
+                                ),
+                                (
+                                    "svd",
+                                    TruncatedSVD(
+                                        algorithm="arpack",
+                                        n_components=max(
+                                            1,
+                                            min(
+                                                num_examples // 10 + 1,
+                                                num_features // 2,
+                                            ),
+                                        ),
+                                        random_state=random_state,
+                                    ),
+                                ),
+                            ],
+                        ),
+                    ),
+                ],
+            ),
+        }
+
+    def __init__(
+        self,
+        *,
+        transform_name: str = "safepower",
+        apply_to_categorical: bool = False,
+        append_to_original: bool | Literal["auto"] = False,
+        subsample_features: float = -1,
+        global_transformer_name: str | None = None,
+        random_state: int | np.random.Generator | None = None,
+    ):
+        super().__init__()
+        self.transform_name = transform_name
+        self.apply_to_categorical = apply_to_categorical
+        self.append_to_original = append_to_original
+        self.random_state = random_state
+        self.subsample_features = float(subsample_features)
+        self.global_transformer_name = global_transformer_name
+        self.transformer_: Pipeline | ColumnTransformer | None = None
+
+    def _set_transformer_and_cat_ix(  # noqa: PLR0912
+        self,
+        n_samples: int,
+        n_features: int,
+        categorical_features: list[int],
+    ) -> tuple[Pipeline | ColumnTransformer, list[int]]:
+        if "adaptive" in self.transform_name:
+            raise NotImplementedError("Adaptive preprocessing raw removed.")
+
+        static_seed, rng = infer_random_state(self.random_state)
+
+        if (
+            self.global_transformer_name is not None
+            and self.global_transformer_name != "None"
+            and not (self.global_transformer_name == "svd" and n_features < 2)
+        ):
+            global_transformer_ = self.get_all_global_transformers(
+                n_samples,
+                n_features,
+                random_state=static_seed,
+            )[self.global_transformer_name]
+        else:
+            global_transformer_ = None
+
+        all_preprocessors = self.get_all_preprocessors(
+            n_samples,
+            random_state=static_seed,
+        )
+        if self.subsample_features > 0:
+            subsample_features = int(self.subsample_features * n_features) + 1
+            # sampling more features than exist
+            replace = subsample_features > n_features
+            self.subsampled_features_ = rng.choice(
+                list(range(n_features)),
+                subsample_features,
+                replace=replace,
+            )
+            categorical_features = [
+                new_idx
+                for new_idx, idx in enumerate(self.subsampled_features_)
+                if idx in categorical_features
+            ]
+            n_features = subsample_features
+        else:
+            self.subsampled_features_ = np.arange(n_features)
+
+        all_feats_ix = list(range(n_features))
+        transformers = []
+
+        numerical_ix = [i for i in range(n_features) if i not in categorical_features]
+
+        append_decision = n_features < self.APPEND_TO_ORIGINAL_THRESHOLD
+        self.append_to_original = (
+            append_decision
+            if self.append_to_original == "auto"
+            else self.append_to_original
+        )
+
+        # -------- Append to original ------
+        # If we append to original, all the categorical indices are kept in place
+        # as the first transform is a passthrough on the whole X as it is above
+        if self.append_to_original and self.apply_to_categorical:
+            trans_ixs = categorical_features + numerical_ix
+            transformers.append(("original", "passthrough", all_feats_ix))
+            cat_ix = categorical_features  # Exist as they are in original
+
+        elif self.append_to_original and not self.apply_to_categorical:
+            trans_ixs = numerical_ix
+            # Includes the categoricals passed through
+            transformers.append(("original", "passthrough", all_feats_ix))
+            cat_ix = categorical_features  # Exist as they are in original
+
+        # -------- Don't append to original ------
+        # We only have categorical indices if we don't transform them
+        # The first transformer will be a passthrough on the categorical indices
+        # Making them the first
+        elif not self.append_to_original and self.apply_to_categorical:
+            trans_ixs = categorical_features + numerical_ix
+            cat_ix = []  # We have none left, they've been transformed
+
+        elif not self.append_to_original and not self.apply_to_categorical:
+            trans_ixs = numerical_ix
+            transformers.append(("cats", "passthrough", categorical_features))
+            cat_ix = list(range(len(categorical_features)))  # They are at start
+
+        else:
+            raise ValueError(
+                f"Unrecognized combination of {self.apply_to_categorical=}"
+                f" and {self.append_to_original=}",
+            )
+
+        # NOTE: No need to keep track of categoricals here, already done above
+        if self.transform_name != "per_feature":
+            _transformer = all_preprocessors[self.transform_name]
+            transformers.append(("feat_transform", _transformer, trans_ixs))
+        else:
+            preprocessors = list(all_preprocessors.values())
+            transformers.extend(
+                [
+                    (f"transformer_{i}", rng.choice(preprocessors), [i])  # type: ignore
+                    for i in trans_ixs
+                ],
+            )
+
+        transformer = ColumnTransformer(
+            transformers,
+            remainder="drop",
+            sparse_threshold=0.0,  # No sparse
+        )
+
+        # Apply a global transformer which accepts the entire dataset instead of
+        # one column
+        # NOTE: We assume global_transformer does not destroy the semantic meaning of
+        # categorical_features_.
+        if global_transformer_:
+            transformer = Pipeline(
+                [
+                    ("preprocess", transformer),
+                    ("global_transformer", global_transformer_),
+                ],
+            )
+
+        self.transformer_ = transformer
+
+        return transformer, cat_ix
+
+    @override
+    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
+        n_samples, n_features = X.shape
+        transformer, cat_ix = self._set_transformer_and_cat_ix(
+            n_samples,
+            n_features,
+            categorical_features,
+        )
+        transformer.fit(X[:, self.subsampled_features_])
+        self.categorical_features_after_transform_ = cat_ix
+        self.transformer_ = transformer
+        return cat_ix
+
+    @override
+    def fit_transform(
+        self,
+        X: np.ndarray,
+        categorical_features: list[int],
+    ) -> _TransformResult:
+        n_samples, n_features = X.shape
+        transformer, cat_ix = self._set_transformer_and_cat_ix(
+            n_samples,
+            n_features,
+            categorical_features,
+        )
+        Xt = transformer.fit_transform(X[:, self.subsampled_features_])
+        self.categorical_features_after_transform_ = cat_ix
+        self.transformer_ = transformer
+        return _TransformResult(Xt, cat_ix)  # type: ignore
+
+    @override
+    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
+        assert self.transformer_ is not None, "You must call fit first"
+        return self.transformer_.transform(X[:, self.subsampled_features_])  # type: ignore
+
+
+class EncodeCategoricalFeaturesStep(FeaturePreprocessingTransformerStep):
+    def __init__(
+        self,
+        categorical_transform_name: str = "ordinal",
+        random_state: int | np.random.Generator | None = None,
+    ):
+        super().__init__()
+        self.categorical_transform_name = categorical_transform_name
+        self.random_state = random_state
+
+        self.categorical_transformer_ = None
+
+    @staticmethod
+    def get_least_common_category_count(x_column: np.ndarray) -> int:
+        if len(x_column) == 0:
+            return 0
+        counts = np.unique(x_column, return_counts=True)[1]
+        return int(counts.min())
+
+    def _get_transformer(
+        self,
+        X: np.ndarray,
+        categorical_features: list[int],
+    ) -> tuple[ColumnTransformer | None, list[int]]:
+        if self.categorical_transform_name.startswith("ordinal"):
+            name = self.categorical_transform_name[len("ordinal") :]
+            # Create a column transformer
+            if name.startswith("_common_categories"):
+                name = name[len("_common_categories") :]
+                categorical_features = [
+                    i
+                    for i, col in enumerate(X.T)
+                    if i in categorical_features
+                    and self.get_least_common_category_count(col) >= 10
+                ]
+            elif name.startswith("_very_common_categories"):
+                name = name[len("_very_common_categories") :]
+                categorical_features = [
+                    i
+                    for i, col in enumerate(X.T)
+                    if i in categorical_features
+                    and self.get_least_common_category_count(col) >= 10
+                    and len(np.unique(col)) < (len(X) // 10)  # type: ignore
+                ]
+
+            assert name in ("_shuffled", ""), (
+                "unknown categorical transform name, should be 'ordinal'"
+                f" or 'ordinal_shuffled' it was {self.categorical_transform_name}"
+            )
+
+            ct = ColumnTransformer(
+                [
+                    (
+                        "ordinal_encoder",
+                        OrdinalEncoder(
+                            handle_unknown="use_encoded_value",
+                            unknown_value=np.nan,
+                        ),  # 'sparse' has been deprecated
+                        categorical_features,
+                    ),
+                ],
+                # The column numbers to be transformed
+                remainder="passthrough",  # Leave the rest of the columns untouched
+            )
+            return ct, categorical_features
+
+        if self.categorical_transform_name == "onehot":
+            # Create a column transformer
+            ct = ColumnTransformer(
+                [
+                    (
+                        "one_hot_encoder",
+                        OneHotEncoder(
+                            drop="if_binary",
+                            sparse_output=False,
+                            handle_unknown="ignore",
+                        ),
+                        categorical_features,
+                    ),
+                ],
+                # The column numbers to be transformed
+                remainder="passthrough",  # Leave the rest of the columns untouched
+            )
+            return ct, categorical_features
+
+        if self.categorical_transform_name in ("numeric", "none"):
+            return None, categorical_features
+        raise ValueError(
+            f"Unknown categorical transform {self.categorical_transform_name}",
+        )
+
+    def _fit(
+        self,
+        X: np.ndarray,
+        categorical_features: list[int],
+    ) -> list[int]:
+        ct, categorical_features = self._get_transformer(X, categorical_features)
+        if ct is None:
+            self.categorical_transformer_ = None
+            return categorical_features
+
+        _, rng = infer_random_state(self.random_state)
+
+        if self.categorical_transform_name.startswith("ordinal"):
+            ct.fit(X)
+            categorical_features = list(range(len(categorical_features)))
+
+            self.random_mappings_ = {}
+            if self.categorical_transform_name.endswith("_shuffled"):
+                for col_ix in categorical_features:
+                    col_cats = len(
+                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
+                    )
+                    perm = rng.permutation(col_cats)
+                    self.random_mappings_[col_ix] = perm
+
+        elif self.categorical_transform_name == "onehot":
+            Xt = ct.fit_transform(X)
+            if Xt.size >= 1_000_000:
+                ct = None
+            else:
+                categorical_features = list(range(Xt.shape[1]))[
+                    ct.output_indices_["one_hot_encoder"]
+                ]
+        else:
+            raise ValueError(
+                f"Unknown categorical transform {self.categorical_transform_name}",
+            )
+
+        self.categorical_transformer_ = ct
+        return categorical_features
+
+    def _fit_transform(
+        self,
+        X: np.ndarray,
+        categorical_features: list[int],
+    ) -> tuple[np.ndarray, list[int]]:
+        ct, categorical_features = self._get_transformer(X, categorical_features)
+        if ct is None:
+            self.categorical_transformer_ = None
+            return X, categorical_features
+
+        _, rng = infer_random_state(self.random_state)
+
+        if self.categorical_transform_name.startswith("ordinal"):
+            Xt = ct.fit_transform(X)
+            categorical_features = list(range(len(categorical_features)))
+
+            self.random_mappings_ = {}
+            if self.categorical_transform_name.endswith("_shuffled"):
+                for col_ix in categorical_features:
+                    col_cats = len(
+                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
+                    )
+                    perm = rng.permutation(col_cats)
+                    self.random_mappings_[col_ix] = perm
+
+                    Xcol: np.ndarray = Xt[:, col_ix]  # type: ignore
+                    not_nan_mask = ~np.isnan(Xcol)
+                    Xcol[not_nan_mask] = perm[Xcol[not_nan_mask].astype(int)].astype(
+                        Xcol.dtype,
+                    )
+
+        elif self.categorical_transform_name == "onehot":
+            Xt = ct.fit_transform(X)
+            if Xt.size >= 1_000_000:
+                ct = None
+                Xt = X
+            else:
+                categorical_features = list(range(Xt.shape[1]))[
+                    ct.output_indices_["one_hot_encoder"]
+                ]
+        else:
+            raise ValueError(
+                f"Unknown categorical transform {self.categorical_transform_name}",
+            )
+
+        self.categorical_transformer_ = ct
+        return Xt, categorical_features  # type: ignore
+
+    @override
+    def fit_transform(
+        self,
+        X: np.ndarray,
+        categorical_features: list[int],
+    ) -> _TransformResult:
+        Xt, cat_ix = self._fit_transform(X, categorical_features)
+        self.categorical_features_after_transform_ = cat_ix
+        return _TransformResult(Xt, cat_ix)
+
+    @override
+    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
+        if self.categorical_transformer_ is None:
+            return X
+
+        import warnings
+
+        with warnings.catch_warnings():
+            warnings.filterwarnings(
+                "ignore", message=".*Found unknown categories in col.*"
+            )  # These warnings are expected when transforming test data
+            transformed = self.categorical_transformer_.transform(X)
+        if self.categorical_transform_name.endswith("_shuffled"):
+            for col, mapping in self.random_mappings_.items():
+                not_nan_mask = ~np.isnan(transformed[:, col])  # type: ignore
+                transformed[:, col][not_nan_mask] = mapping[
+                    transformed[:, col][not_nan_mask].astype(int)
+                ].astype(transformed[:, col].dtype)
+        return transformed  # type: ignore
+
+
+class NanHandlingPolynomialFeaturesStep(FeaturePreprocessingTransformerStep):
+    def __init__(
+        self,
+        *,
+        max_features: int | None = None,
+        random_state: int | np.random.Generator | None = None,
+    ):
+        super().__init__()
+
+        self.max_poly_features = max_features
+        self.random_state = random_state
+
+        self.poly_factor_1_idx: np.ndarray | None = None
+        self.poly_factor_2_idx: np.ndarray | None = None
+
+        self.standardizer = StandardScaler(with_mean=False)
+
+    @override
+    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
+        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"
+        _, rng = infer_random_state(self.random_state)
+
+        if X.shape[0] == 0 or X.shape[1] == 0:
+            return [*categorical_features]
+
+        # How many polynomials can we create?
+        n_polynomials = (X.shape[1] * (X.shape[1] - 1)) // 2 + X.shape[1]
+        n_polynomials = (
+            min(self.max_poly_features, n_polynomials)
+            if self.max_poly_features
+            else n_polynomials
+        )
+
+        X = self.standardizer.fit_transform(X)
+
+        # Randomly select the indices of the factors
+        self.poly_factor_1_idx = rng.choice(
+            np.arange(0, X.shape[1]),
+            size=n_polynomials,
+            replace=True,
+        )
+        self.poly_factor_2_idx = np.ones_like(self.poly_factor_1_idx) * -1
+        for i in range(len(self.poly_factor_1_idx)):
+            while self.poly_factor_2_idx[i] == -1:
+                poly_factor_1_ = self.poly_factor_1_idx[i]
+                # indices of the factors that have already been used
+                used_indices = self.poly_factor_2_idx[
+                    self.poly_factor_1_idx == poly_factor_1_
+                ]
+                # remaining indices, only factors with higher index can be selected
+                # to avoid duplicates
+                indices_ = set(range(poly_factor_1_, X.shape[1])) - set(
+                    used_indices.tolist(),
+                )
+                if len(indices_) == 0:
+                    self.poly_factor_1_idx[i] = rng.choice(
+                        np.arange(0, X.shape[1]),
+                        size=1,
+                    )
+                    continue
+                self.poly_factor_2_idx[i] = rng.choice(list(indices_), size=1)
+
+        return categorical_features
+
+    @override
+    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
+        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"
+
+        if X.shape[0] == 0 or X.shape[1] == 0:
+            return X
+
+        X = self.standardizer.transform(X)  # type: ignore
+
+        poly_features_xs = X[:, self.poly_factor_1_idx] * X[:, self.poly_factor_2_idx]
+
+        return np.hstack((X, poly_features_xs))
+
+
+class DifferentiableZNormStep(FeaturePreprocessingTransformerStep):
+    def __init__(self):
+        super().__init__()
+
+        self.means = torch.tensor([])
+        self.stds = torch.tensor([])
+
+    def _fit(self, X: torch.Tensor, categorical_features: list[int]) -> list[int]:
+        self.means = X.mean(dim=0, keepdim=True)
+        self.stds = X.std(dim=0, keepdim=True)
+        return categorical_features
+
+    def _transform(self, X: torch.Tensor, *, is_test=False):  # noqa: ARG002
+        assert X.shape[1] == self.means.shape[1]
+        assert X.shape[1] == self.stds.shape[1]
+        return (X - self.means) / self.stds
diff --git a/src/tabpfn/architectures/base/tabpfn_col_embedding.pt b/src/tabpfn/architectures/base/tabpfn_col_embedding.pt
deleted file mode 100644
index 95f75bc..0000000
Binary files a/src/tabpfn/architectures/base/tabpfn_col_embedding.pt and /dev/null differ
diff --git a/src/tabpfn/architectures/base/thinking_tokens.py b/src/tabpfn/architectures/base/thinking_tokens.py
deleted file mode 100644
index ad3a2c8..0000000
--- a/src/tabpfn/architectures/base/thinking_tokens.py
+++ /dev/null
@@ -1,60 +0,0 @@
-from __future__ import annotations
-
-from typing_extensions import override
-
-import torch
-from torch import Tensor
-from torch.nn import Module, Parameter
-
-
-class AddThinkingTokens(Module):
-    """Takes the embedded input and prepends "thinking tokens" to it.
-
-    Adjusts the single_eval_pos appropriately to account for the new, longer input.
-
-    We hope that the thinking tokens give the model more computational capacity to
-    perform in-context learning, particuarly on small datasets. This is inspired by LLM
-    results such as
-    - Think Before You Speak, Goyal et al. 2024:
-        https://openreview.net/forum?id=ph04CRkPdC
-    - Exact Expressive Power of Transformers with Padding, Merrill & Sabharwal 2025:
-        https://arxiv.org/abs/2505.18948
-    """
-
-    def __init__(self, num_thinking_rows: int, emsize: int) -> None:
-        super().__init__()
-        self.num_thinking_rows = num_thinking_rows
-        # We have to work with variable numbers of features, so we use the same token
-        # for each feature.
-        self.row_token_values = Parameter(torch.empty(num_thinking_rows, emsize))
-        self.reset_parameters()
-
-    @override
-    def forward(
-        self, embedded_input: Tensor, single_eval_pos: int
-    ) -> tuple[Tensor, int]:
-        """Prepends the thinking tokens to the embedded input.
-
-        Args:
-            embedded_input: [batch x train+eval rows x feature groups x emsize]
-            single_eval_pos: Rows after this index are treated as evaluation rows.
-
-        Returns:
-            (
-                embedded_input with added rows
-                    [batch size x thinking+train+eval rows x feature groups x emsize],
-                updated single_eval_pos
-            )
-        """
-        batch_size, _, num_features, _ = embedded_input.shape
-        thinking_tokens_base = self.row_token_values.unsqueeze(0).unsqueeze(2)
-        thinking_tokens = thinking_tokens_base.expand(batch_size, -1, num_features, -1)
-
-        embedded_input = torch.cat([thinking_tokens, embedded_input], dim=1)
-        single_eval_pos += self.num_thinking_rows
-        return embedded_input, single_eval_pos
-
-    def reset_parameters(self) -> None:
-        # This is the initialisation used in torch.nn.Embedding, so hopefully a
-        # reasonable choice for our application.
-        torch.nn.init.normal_(self.row_token_values)
diff --git a/src/tabpfn/architectures/base/transformer.py b/src/tabpfn/architectures/base/transformer.py
index 38b7d39..8d2e525 100644
--- a/src/tabpfn/architectures/base/transformer.py
+++ b/src/tabpfn/architectures/base/transformer.py
@@ -2,11 +2,10 @@
 
 from __future__ import annotations
 
-import logging
 import warnings
-from collections.abc import Callable, Iterable
+from collections.abc import Callable, Generator, Iterable
+from contextlib import contextmanager
 from functools import partial
-from pathlib import Path
 from typing import TYPE_CHECKING, Any, Literal, overload
 from typing_extensions import Self, override
 
@@ -23,19 +22,24 @@ from tabpfn.architectures.base.encoders import (
     SequentialEncoder,
 )
 from tabpfn.architectures.base.layer import PerFeatureEncoderLayer
-from tabpfn.architectures.base.thinking_tokens import AddThinkingTokens
 from tabpfn.architectures.interface import Architecture
 
 if TYPE_CHECKING:
     from tabpfn.architectures.base.config import ModelConfig
 
 
-logger = logging.getLogger(__name__)
-
-# Hard coded "random" embeddings (seed=42) used during training of size
-# 2000 x 48.
-col_embedding_path = Path(__file__).parent / "tabpfn_col_embedding.pt"
-COL_EMBEDDING = torch.load(col_embedding_path, weights_only=True)
+@contextmanager
+def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
+    torch_rng_state = torch.get_rng_state()
+    if torch.cuda.is_available():
+        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
+    torch.manual_seed(seed)
+    try:
+        yield
+    finally:
+        torch.set_rng_state(torch_rng_state)
+        if torch.cuda.is_available():
+            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)
 
 
 class LayerStack(nn.Module):
@@ -45,6 +49,7 @@ class LayerStack(nn.Module):
         self,
         *,
         layers: Iterable[nn.Module],
+        recompute_each_layer: bool,
         min_num_layers_layer_dropout: int | None,
     ) -> None:
         super().__init__()
@@ -54,6 +59,7 @@ class LayerStack(nn.Module):
             if min_num_layers_layer_dropout is not None
             else len(self.layers)
         )
+        self.recompute_each_layer = recompute_each_layer
 
     @classmethod
     def of_repeated_layer(
@@ -61,11 +67,13 @@ class LayerStack(nn.Module):
         layer_creator: Callable[[], nn.Module],
         *,
         num_layers: int,
+        recompute_each_layer: bool = False,
         min_num_layers_layer_dropout: int | None = None,
     ) -> Self:
         """Returns an instance containing the given layer repeated num_layers times."""
         return cls(
             layers=[layer_creator() for _ in range(num_layers)],
+            recompute_each_layer=recompute_each_layer,
             min_num_layers_layer_dropout=min_num_layers_layer_dropout,
         )
 
@@ -73,7 +81,6 @@ class LayerStack(nn.Module):
     def forward(
         self,
         x: torch.Tensor,
-        recompute_layer: bool,
         **kwargs: Any,
     ) -> torch.Tensor:
         n_layers = torch.randint(
@@ -81,7 +88,7 @@ class LayerStack(nn.Module):
         ).item()
 
         for layer in self.layers[:n_layers]:
-            if recompute_layer and x.requires_grad:
+            if self.recompute_each_layer and x.requires_grad:
                 x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
             else:
                 x = layer(x, **kwargs)
@@ -186,14 +193,6 @@ class PerFeatureTransformer(Architecture):
         self.cache_trainset_representation = cache_trainset_representation
         self.cached_embeddings: torch.Tensor | None = None
 
-        if config.num_thinking_rows > 0:
-            self.add_thinking_tokens = AddThinkingTokens(
-                num_thinking_rows=config.num_thinking_rows,
-                emsize=config.emsize,
-            )
-        else:
-            self.add_thinking_tokens = None
-
         layer_creator = lambda: PerFeatureEncoderLayer(
             config=config,
             dim_feedforward=nhid,
@@ -205,11 +204,10 @@ class PerFeatureTransformer(Architecture):
             **layer_kwargs,
         )
 
-        self.recompute_layer = config.recompute_layer
-
         self.transformer_encoder = LayerStack.of_repeated_layer(
             layer_creator=layer_creator,
             num_layers=config.nlayers,
+            recompute_each_layer=config.recompute_layer,
             min_num_layers_layer_dropout=min_num_layers_layer_dropout,
         )
 
@@ -263,7 +261,7 @@ class PerFeatureTransformer(Architecture):
 
         self.dag_pos_enc_dim = config.dag_pos_enc_dim
         self.cached_feature_positional_embeddings: torch.Tensor | None = None
-        self.random_embedding_seed = config.seed
+        self.seed = config.seed
 
     def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
         """Sets the save_peak_mem_factor for all layers.
@@ -307,7 +305,6 @@ class PerFeatureTransformer(Architecture):
         categorical_inds: list[list[int]] | None = None,
         style: torch.Tensor | None = None,
         data_dags: list[nx.DiGraph] | None = None,
-        force_recompute_layer: bool = False,
     ) -> torch.Tensor: ...
 
     @overload
@@ -320,7 +317,6 @@ class PerFeatureTransformer(Architecture):
         categorical_inds: list[list[int]] | None = None,
         style: torch.Tensor | None = None,
         data_dags: list[nx.DiGraph] | None = None,
-        force_recompute_layer: bool = False,
     ) -> dict[str, torch.Tensor]: ...
 
     @override
@@ -333,7 +329,6 @@ class PerFeatureTransformer(Architecture):
         categorical_inds: list[list[int]] | None = None,
         style: torch.Tensor | None = None,
         data_dags: list[nx.DiGraph] | None = None,
-        force_recompute_layer: bool = False,
     ) -> torch.Tensor | dict[str, torch.Tensor]:
         """Perform a forward pass.
 
@@ -358,8 +353,6 @@ class PerFeatureTransformer(Architecture):
             categorical_inds: The indices of categorical features.
             style: The style vector.
             data_dags: The data DAGs for each example in the batch.
-            force_recompute_layer: Whether to force to recompute layers (i.e.
-            perform activation checkpointing for each layer).
         """
         assert style is None
 
@@ -553,13 +546,6 @@ class PerFeatureTransformer(Architecture):
             )
         del embedded_y, embedded_x
 
-        if self.add_thinking_tokens is not None:
-            embedded_input, single_eval_pos = self.add_thinking_tokens(
-                embedded_input,
-                single_eval_pos,
-            )
-
-        recompute_layer = self.recompute_layer or force_recompute_layer
         encoder_out = self.transformer_encoder(
             (
                 embedded_input
@@ -568,7 +554,6 @@ class PerFeatureTransformer(Architecture):
             ),
             single_eval_pos=single_eval_pos,
             cache_trainset_representation=self.cache_trainset_representation,
-            recompute_layer=recompute_layer,
         )  # b s f+1 e -> b s f+1 e
 
         # If we are using a decoder
@@ -607,14 +592,7 @@ class PerFeatureTransformer(Architecture):
             )
 
             # out: s b e
-            thinking_rows_offset = (
-                self.add_thinking_tokens.num_thinking_rows
-                if self.add_thinking_tokens is not None
-                else 0
-            )
-            train_encoder_out = encoder_out[
-                :, thinking_rows_offset:single_eval_pos, -1
-            ].transpose(0, 1)
+            train_encoder_out = encoder_out[:, :single_eval_pos, -1].transpose(0, 1)
             output_decoded["train_embeddings"] = train_encoder_out
             output_decoded["test_embeddings"] = test_encoder_out
 
@@ -632,82 +610,61 @@ class PerFeatureTransformer(Architecture):
         use_cached_embeddings: bool = False,
     ) -> tuple[torch.Tensor, torch.Tensor]:
         if use_cached_embeddings and self.cached_embeddings is not None:
-            msg = "Caching embeddings is not supported with data_dags at this point."
-            assert data_dags is None, msg
+            assert (
+                data_dags is None
+            ), "Caching embeddings is not supported with data_dags at this point."
             x += self.cached_embeddings[None, None]
             return x, y
 
-        if torch.jit.is_tracing():
-            # jit tracing is used during onnx export, but does not support tracing the
-            # Generator below. This means that the model will use different random
-            # positional embeddings than during training, which will decrease the
-            # quality of the predictions.
-            logger.warning(
-                "TabPFN does not fully support exporting the model using tracing. "
-                "The exported model may work, but will give lower quality predictions."
-            )
-            positional_embedding_rng = None
-        else:
-            positional_embedding_rng = torch.Generator(device=x.device).manual_seed(
-                self.random_embedding_seed
-            )
-
-        if self.feature_positional_embedding == "normal_rand_vec":
-            embs = torch.randn(
-                (x.shape[2], x.shape[3]),
-                device=x.device,
-                dtype=x.dtype,
-                generator=positional_embedding_rng,
-            )
-            x += embs[None, None]
-        elif self.feature_positional_embedding == "uni_rand_vec":
-            embs = (
-                torch.rand(
+        # TODO: we should probably hardcode the seed here
+        # I think we never want to change it?
+        with isolate_torch_rng(self.seed, device=x.device):
+            if self.feature_positional_embedding == "normal_rand_vec":
+                embs = torch.randn(
                     (x.shape[2], x.shape[3]),
                     device=x.device,
                     dtype=x.dtype,
-                    generator=positional_embedding_rng,
                 )
-                * 2
-                - 1
-            )
-            x += embs[None, None]
-        elif self.feature_positional_embedding == "learned":
-            w = self.feature_positional_embedding_embeddings.weight
-            embs = w[
-                torch.randint(
-                    0,
-                    w.shape[0],
-                    (x.shape[2],),
-                    generator=positional_embedding_rng,
+                x += embs[None, None]
+            elif self.feature_positional_embedding == "uni_rand_vec":
+                embs = (
+                    torch.rand(
+                        (x.shape[2], x.shape[3]),
+                        device=x.device,
+                        dtype=x.dtype,
+                    )
+                    * 2
+                    - 1
                 )
-            ]
-            x += embs[None, None]
-        elif self.feature_positional_embedding == "subspace":
-            embs = torch.randn(
-                (x.shape[2], x.shape[3] // 4),
-                device=x.device,
-                dtype=x.dtype,
-                generator=positional_embedding_rng,
-            )
-            # Random numbers on CPU and GPU are different. We fixed the seed, so these
-            # are not actually random, leading to a performance drop on CPU without
-            # hardcoding them.
-            if embs.shape[1] == 48 and self.random_embedding_seed == 42:  # 192 // 4
-                embs[:2000] = COL_EMBEDDING[: embs.shape[0]].to(
-                    device=embs.device, dtype=embs.dtype
+                x += embs[None, None]
+            elif self.feature_positional_embedding == "learned":
+                w = self.feature_positional_embedding_embeddings.weight
+                embs = w[
+                    torch.randint(
+                        0,
+                        w.shape[0],
+                        (x.shape[2],),
+                    )
+                ]
+                x += embs[None, None]
+            elif self.feature_positional_embedding == "subspace":
+                embs = torch.randn(
+                    (x.shape[2], x.shape[3] // 4),
+                    device=x.device,
+                    dtype=x.dtype,
                 )
-            embs = self.feature_positional_embedding_embeddings(embs)
-            x += embs[None, None]
-        elif self.feature_positional_embedding is None:
-            embs = None
-        else:
-            raise ValueError(f"Unknown {self.feature_positional_embedding=}")
+                embs = self.feature_positional_embedding_embeddings(embs)
+                x += embs[None, None]
+            elif self.feature_positional_embedding is None:
+                embs = None
+            else:
+                raise ValueError(f"Unknown {self.feature_positional_embedding=}")
 
         self.cached_embeddings = None
         if cache_embeddings and embs is not None:
-            msg = "Caching embeddings is not supported with data_dags at this point."
-            assert data_dags is None, msg
+            assert (
+                data_dags is None
+            ), "Caching embeddings is not supported with data_dags at this point."
             self.cached_embeddings = embs
 
         # TODO(old) should this go into encoder?
@@ -772,6 +729,28 @@ class PerFeatureTransformer(Architecture):
         for layer in (self.transformer_decoder or self.transformer_encoder).layers:
             layer.empty_trainset_representation_cache()
 
+    def _transform_categorical_indices_feat_groups(
+        self, categorical_inds: list[int], n_subgroups: int
+    ) -> list[list[int]]:
+        """Transform the categorical indices list(s)
+        to align with the feature groups.
+
+        Args:
+            categorical_inds: categorical indices as 2D list
+            n_subgroups: number of subgroups.
+        """
+        new_categorical_inds = []
+        for subgroup in range(n_subgroups):
+            subgroup_lower = subgroup * self.features_per_group
+            subgroup_upper = (subgroup + 1) * self.features_per_group
+            subgroup_indices = [
+                i - subgroup_lower
+                for i in categorical_inds
+                if subgroup_lower <= i < subgroup_upper
+            ]
+            new_categorical_inds.append(subgroup_indices)
+        return new_categorical_inds
+
 
 def _networkx_add_direct_connections(graph: nx.DiGraph) -> bool:
     added_connection = False
@@ -805,9 +784,7 @@ def _add_pos_emb(
     is_undirected: bool = False,
     k: int = 20,
 ) -> None:
-    # Local import because scipy is quite heavy and the graph embeddings are not used by
-    # default.
-    from scipy.sparse.linalg import eigs, eigsh  # noqa: PLC0415
+    from scipy.sparse.linalg import eigs, eigsh
 
     eig_fn = eigs if not is_undirected else eigsh
 
diff --git a/src/tabpfn/base.py b/src/tabpfn/base.py
index 32c42e3..1b66b2a 100644
--- a/src/tabpfn/base.py
+++ b/src/tabpfn/base.py
@@ -4,20 +4,19 @@
 
 from __future__ import annotations
 
-import pathlib
 import warnings
-from collections.abc import Sequence
-from typing import TYPE_CHECKING, Any, Callable, Literal, Union
+from pathlib import Path
+from typing import TYPE_CHECKING, Any, Callable, Literal, Union, overload
 
 import torch
 
 from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
+from tabpfn.config import ModelInterfaceConfig
 
 # --- TabPFN imports ---
 from tabpfn.constants import (
     AUTOCAST_DTYPE_BYTE_SIZE,
     DEFAULT_DTYPE_BYTE_SIZE,
-    ModelPath,
     XType,
     YType,
 )
@@ -28,7 +27,7 @@ from tabpfn.inference import (
     InferenceEngineCachePreprocessing,
     InferenceEngineOnDemand,
 )
-from tabpfn.model_loading import load_model_criterion_config, resolve_model_version
+from tabpfn.model_loading import load_model_criterion_config
 from tabpfn.preprocessing import (
     BaseDatasetConfig,
     ClassifierDatasetConfig,
@@ -37,7 +36,7 @@ from tabpfn.preprocessing import (
 )
 from tabpfn.settings import settings
 from tabpfn.utils import (
-    infer_devices,
+    infer_device_and_type,
     infer_fp16_inference_mode,
     infer_random_state,
     split_large_data,
@@ -50,23 +49,14 @@ if TYPE_CHECKING:
 
     from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
     from tabpfn.architectures.interface import Architecture, ArchitectureConfig
-    from tabpfn.classifier import TabPFNClassifier
-    from tabpfn.inference_config import InferenceConfig
-    from tabpfn.regressor import TabPFNRegressor
 
 
 class BaseModelSpecs:
     """Base class for model specifications."""
 
-    def __init__(
-        self,
-        model: Architecture,
-        architecture_config: ArchitectureConfig,
-        inference_config: InferenceConfig,
-    ):
+    def __init__(self, model: Architecture, config: ArchitectureConfig):
         self.model = model
-        self.architecture_config = architecture_config
-        self.inference_config = inference_config
+        self.config = config
 
 
 class ClassifierModelSpecs(BaseModelSpecs):
@@ -81,159 +71,106 @@ class RegressorModelSpecs(BaseModelSpecs):
     def __init__(
         self,
         model: Architecture,
-        architecture_config: ArchitectureConfig,
-        inference_config: InferenceConfig,
+        config: ArchitectureConfig,
         norm_criterion: FullSupportBarDistribution,
     ):
-        super().__init__(model, architecture_config, inference_config)
+        super().__init__(model, config)
         self.norm_criterion = norm_criterion
 
 
 ModelSpecs = Union[RegressorModelSpecs, ClassifierModelSpecs]
 
 
+@overload
+def initialize_tabpfn_model(
+    model_path: (str | Path | Literal["auto"] | RegressorModelSpecs),
+    which: Literal["regressor"],
+    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
+) -> RegressorModelSpecs: ...
+
+
+@overload
+def initialize_tabpfn_model(
+    model_path: (str | Path | Literal["auto"] | ClassifierModelSpecs),
+    which: Literal["classifier"],
+    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
+) -> ClassifierModelSpecs: ...
+
+
 def initialize_tabpfn_model(
-    model_path: ModelPath
-    | list[ModelPath]
+    model_path: str
+    | Path
+    | Literal["auto"]
     | RegressorModelSpecs
-    | ClassifierModelSpecs
-    | list[RegressorModelSpecs]
-    | list[ClassifierModelSpecs],
+    | ClassifierModelSpecs,
     which: Literal["classifier", "regressor"],
     fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
-) -> tuple[
-    list[Architecture],
-    list[ArchitectureConfig],
-    FullSupportBarDistribution | None,
-    InferenceConfig,
-]:
+) -> ModelSpecs:
     """Initializes a TabPFN model based on the provided configuration.
 
     Args:
         model_path: Path or directive ("auto") to load the pre-trained model from.
-            If a list of paths is provided, the models are applied across different
-            estimators. If a RegressorModelSpecs or ClassifierModelSpecs object is
-            provided, the model is loaded from the object.
-
         which: Which TabPFN model to load.
         fit_mode: Determines caching behavior.
 
     Returns:
-        a list of models,
-        a list of architecture configs (associated with each model),
-        if regression, the bar distribution, otherwise None,
-        the inference config
+        model: The loaded TabPFN model.
+        config: The configuration object associated with the loaded model.
+        bar_distribution: The BarDistribution for regression (`None` if classifier).
     """
+    model, config, norm_criterion = None, None, None
     if isinstance(model_path, RegressorModelSpecs) and which == "regressor":
-        return (
-            [model_path.model],
-            [model_path.architecture_config],
-            model_path.norm_criterion,
-            model_path.inference_config,
-        )
-
-    if isinstance(model_path, ClassifierModelSpecs) and which == "classifier":
-        return (
-            [model_path.model],
-            [model_path.architecture_config],
-            None,
-            model_path.inference_config,
-        )
-
-    if (
-        isinstance(model_path, list)
-        and len(model_path) > 0
-        and all(isinstance(spec, RegressorModelSpecs) for spec in model_path)
-    ):
-        _assert_inference_configs_equal(model_path)
-        return (  # pyright: ignore[reportReturnType]
-            [spec.model for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
-            [spec.architecture_config for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
-            model_path[0].norm_criterion,  # pyright: ignore[reportAttributeAccessIssue]
-            model_path[0].inference_config,
-        )
-
-    if (
-        isinstance(model_path, list)
-        and len(model_path) > 0
-        and all(isinstance(spec, ClassifierModelSpecs) for spec in model_path)
-    ):
-        _assert_inference_configs_equal(model_path)
-        return (
-            [spec.model for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
-            [spec.architecture_config for spec in model_path],  # pyright: ignore[reportAttributeAccessIssue]
-            None,
-            model_path[0].inference_config,
-        )
-
-    if (
-        model_path is None
-        or model_path == "auto"
-        or isinstance(model_path, (str, pathlib.Path, list))  # pyright: ignore[reportArgumentType]
-    ):
-        if isinstance(model_path, list) and len(model_path) == 0:
-            raise ValueError(
-                "You provided a list of model paths with no entries. "
-                "Please provide a valid `model_path` argument, or use 'auto' to use "
-                "the default model."
-            )
-
+        model = model_path.model
+        config = model_path.config
+        norm_criterion = model_path.norm_criterion
+    elif isinstance(model_path, ClassifierModelSpecs) and which == "classifier":
+        model = model_path.model
+        config = model_path.config
+    elif model_path is None or isinstance(model_path, (str, Path)):
+        # (after processing 'auto')
+        download = True
         if isinstance(model_path, str) and model_path == "auto":
             model_path = None  # type: ignore
 
-        version = resolve_model_version(model_path)  # type: ignore
-        download_if_not_exists = True
-
+        # Load model with potential caching
         if which == "classifier":
-            models, _, architecture_configs, inference_config = (
-                load_model_criterion_config(
-                    model_path=model_path,  # pyright: ignore[reportArgumentType]
-                    # The classifier's bar distribution is not used
-                    check_bar_distribution_criterion=False,
-                    cache_trainset_representation=(fit_mode == "fit_with_cache"),
-                    which="classifier",
-                    version=version.value,
-                    download_if_not_exists=download_if_not_exists,
-                )
+            # The classifier's bar distribution is not used;
+            # pass check_bar_distribution_criterion=False
+            model, _, config = load_model_criterion_config(
+                model_path=model_path,
+                check_bar_distribution_criterion=False,
+                cache_trainset_representation=(fit_mode == "fit_with_cache"),
+                which="classifier",
+                version="v2",
+                download=download,
             )
             norm_criterion = None
         else:
-            models, bardist, architecture_configs, inference_config = (
-                load_model_criterion_config(
-                    model_path=model_path,  # pyright: ignore[reportArgumentType]
-                    # The regressor's bar distribution is required
-                    check_bar_distribution_criterion=True,
-                    cache_trainset_representation=(fit_mode == "fit_with_cache"),
-                    which="regressor",
-                    version=version.value,
-                    download_if_not_exists=download_if_not_exists,
-                )
+            # The regressor's bar distribution is required
+            model, bardist, config = load_model_criterion_config(
+                model_path=model_path,
+                check_bar_distribution_criterion=True,
+                cache_trainset_representation=(fit_mode == "fit_with_cache"),
+                which="regressor",
+                version="v2",
+                download=download,
             )
             norm_criterion = bardist
+    else:
+        raise TypeError(
+            "Received ModelSpecs via 'model_path', but 'which' parameter is set to '"
+            + which
+            + "'. Expected 'classifier' or 'regressor'. and model_path"
+            + "is of of type"
+            + str(type(model_path))
+        )
 
-        return models, architecture_configs, norm_criterion, inference_config
-
-    raise TypeError(
-        "Received ModelSpecs via 'model_path', but 'which' parameter is set to '"
-        + which
-        + "'. Expected 'classifier' or 'regressor'. and model_path"
-        + "is of of type"
-        + str(type(model_path))
-    )
-
-
-def _assert_inference_configs_equal(
-    model_specs: list[ClassifierModelSpecs] | list[RegressorModelSpecs],
-) -> None:
-    if not all(
-        spec.inference_config == model_specs[0].inference_config for spec in model_specs
-    ):
-        raise ValueError("All models must have the same inference config")
+    return model, config, norm_criterion
 
 
 def determine_precision(
     inference_precision: torch.dtype | Literal["autocast", "auto"],
-    devices_: Sequence[torch.device],
+    device_: torch.device,
 ) -> tuple[bool, torch.dtype | None, int]:
     """Decide whether to use autocast or a forced precision dtype.
 
@@ -244,7 +181,7 @@ def determine_precision(
             - If `"autocast"`, explicitly use PyTorch autocast (mixed precision).
             - If a `torch.dtype`, force that precision.
 
-        devices_: The devices which will be used for inference.
+        device_: The device on which inference is run.
 
     Returns:
         use_autocast_:
@@ -256,7 +193,7 @@ def determine_precision(
     """
     if inference_precision in ["autocast", "auto"]:
         use_autocast_ = infer_fp16_inference_mode(
-            devices=devices_,
+            device=device_,
             enable=True if (inference_precision == "autocast") else None,
         )
         forced_inference_dtype_ = None
@@ -277,13 +214,13 @@ def create_inference_engine(  # noqa: PLR0913
     *,
     X_train: np.ndarray,
     y_train: np.ndarray,
-    models: list[Architecture],
+    model: Architecture,
     ensemble_configs: Any,
     cat_ix: list[int],
     fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
-    devices_: Sequence[torch.device],
+    device_: torch.device,
     rng: np.random.Generator,
-    n_preprocessing_jobs: int,
+    n_jobs: int,
     byte_size: int,
     forced_inference_dtype_: torch.dtype | None,
     memory_saving_mode: bool | Literal["auto"] | float | int,
@@ -300,14 +237,13 @@ def create_inference_engine(  # noqa: PLR0913
     Args:
         X_train: Training features
         y_train: Training target
-        models: The loaded TabPFN models.
+        model: The loaded TabPFN model.
         ensemble_configs: The ensemble configurations to create multiple "prompts".
         cat_ix: Indices of inferred categorical features.
         fit_mode: Determines how we prepare inference (pre-cache or not).
-        devices_: The devices for inference.
+        device_: The device for inference.
         rng: Numpy random generator.
-        n_preprocessing_jobs: Number of parallel CPU workers to use for the
-            preprocessing.
+        n_jobs: Number of parallel CPU workers.
         byte_size: Byte size for the chosen inference precision.
         forced_inference_dtype_: If not None, the forced dtype for inference.
         memory_saving_mode: GPU/CPU memory saving settings.
@@ -328,8 +264,8 @@ def create_inference_engine(  # noqa: PLR0913
             cat_ix=cat_ix,
             ensemble_configs=ensemble_configs,
             rng=rng,
-            models=models,
-            n_preprocessing_jobs=n_preprocessing_jobs,
+            model=model,
+            n_workers=n_jobs,
             dtype_byte_size=byte_size,
             force_inference_dtype=forced_inference_dtype_,
             save_peak_mem=memory_saving_mode,
@@ -340,8 +276,8 @@ def create_inference_engine(  # noqa: PLR0913
             y_train=y_train,
             cat_ix=cat_ix,
             ensemble_configs=ensemble_configs,
-            models=models,
-            n_preprocessing_jobs=n_preprocessing_jobs,
+            n_workers=n_jobs,
+            model=model,
             rng=rng,
             dtype_byte_size=byte_size,
             force_inference_dtype=forced_inference_dtype_,
@@ -353,10 +289,10 @@ def create_inference_engine(  # noqa: PLR0913
             X_train=X_train,
             y_train=y_train,
             cat_ix=cat_ix,
-            models=models,
+            model=model,
             ensemble_configs=ensemble_configs,
-            n_preprocessing_jobs=n_preprocessing_jobs,
-            devices=devices_,
+            n_workers=n_jobs,
+            device=device_,
             dtype_byte_size=byte_size,
             rng=rng,
             force_inference_dtype=forced_inference_dtype_,
@@ -368,7 +304,7 @@ def create_inference_engine(  # noqa: PLR0913
             X_trains=X_train,
             y_trains=y_train,
             cat_ix=cat_ix,
-            models=models,
+            model=model,
             ensemble_configs=ensemble_configs,
             force_inference_dtype=forced_inference_dtype_,
             inference_mode=inference_mode,
@@ -382,7 +318,7 @@ def create_inference_engine(  # noqa: PLR0913
 
 
 def check_cpu_warning(
-    devices: Sequence[torch.device],
+    device: str | torch.device,
     X: np.ndarray | torch.Tensor | pd.DataFrame,
     *,
     allow_cpu_override: bool = False,
@@ -390,7 +326,7 @@ def check_cpu_warning(
     """Check if using CPU with large datasets and warn or error appropriately.
 
     Args:
-        devices: The torch devices being used
+        device: The torch device being used
         X: The input data (NumPy array, Pandas DataFrame, or Torch Tensor)
         allow_cpu_override: If True, allow CPU usage with large datasets.
     """
@@ -399,13 +335,15 @@ def check_cpu_warning(
     if allow_cpu_override:
         return
 
+    device_mapped = infer_device_and_type(device)
+
     # Determine number of samples
     try:
         num_samples = X.shape[0]
     except AttributeError:
         return
 
-    if any(device.type == "cpu" for device in devices):
+    if torch.device(device_mapped).type == "cpu":
         if num_samples > 1000:
             raise RuntimeError(
                 "Running on CPU with more than 1000 samples is not allowed "
@@ -432,8 +370,6 @@ def get_preprocessed_datasets_helper(
     split_fn: Callable,
     max_data_size: int | None,
     model_type: Literal["regressor", "classifier"],
-    *,
-    equal_split_size: bool,
 ) -> DatasetCollectionWithPreprocessing:
     """Helper function to create a DatasetCollectionWithPreprocessing.
     Relies on methods from the calling_instance for specific initializations.
@@ -447,11 +383,6 @@ def get_preprocessed_datasets_helper(
         max_data_size: Maximum allowed number of samples within one dataset.
         If None, datasets are not splitted.
         model_type: The type of the model.
-        equal_split_size: If True, splits data into equally sized chunks under
-            max_data_size.
-            If False, splits into chunks of size `max_data_size`, with
-            the last chunk having the remainder samples but is dropped if its
-            size is less than 2.
     """
     if not isinstance(X_raw, list):
         X_raw = [X_raw]
@@ -459,17 +390,15 @@ def get_preprocessed_datasets_helper(
         y_raw = [y_raw]
     assert len(X_raw) == len(y_raw), "X and y lists must have the same length."
 
-    if not hasattr(calling_instance, "models_") or calling_instance.models_ is None:
+    if not hasattr(calling_instance, "model_") or calling_instance.model_ is None:
         _, rng = calling_instance._initialize_model_variables()
     else:
-        _static_seed, rng = infer_random_state(calling_instance.random_state)
+        static_seed, rng = infer_random_state(calling_instance.random_state)
 
     X_split, y_split = [], []
     for X_item, y_item in zip(X_raw, y_raw):
         if max_data_size is not None:
-            Xparts, yparts = split_large_data(
-                X_item, y_item, max_data_size, equal_split_size=equal_split_size
-            )
+            Xparts, yparts = split_large_data(X_item, y_item, max_data_size)
         else:
             Xparts, yparts = [X_item], [y_item]
         X_split.extend(Xparts)
@@ -499,7 +428,7 @@ def get_preprocessed_datasets_helper(
                 X_raw=X_mod,
                 y_raw=y_mod,
                 cat_ix=current_cat_ix,
-                znorm_space_bardist_=bardist_,
+                bardist_=bardist_,
             )
         else:
             raise ValueError(f"Invalid model_type: {model_type}")
@@ -509,51 +438,58 @@ def get_preprocessed_datasets_helper(
     return DatasetCollectionWithPreprocessing(split_fn, rng, dataset_config_collection)
 
 
-def initialize_model_variables_helper(
-    calling_instance: TabPFNRegressor | TabPFNClassifier,
+def _initialize_model_variables_helper(
+    calling_instance: Any,
     model_type: Literal["regressor", "classifier"],
 ) -> tuple[int, np.random.Generator]:
-    """Set attributes on the given model to prepare it for inference.
-
-    This includes selecting the device and the inference precision.
-
-    Returns:
-        a tuple (byte_size, rng), where byte_size is the number of bytes in the selected
-        dtype, and rng is a NumPy random Generator for use during inference.
+    """Helper function to perform initialization
+    of the model, return determined byte_size
+    and RNG object.
     """
     static_seed, rng = infer_random_state(calling_instance.random_state)
-    models, architecture_configs, maybe_bardist, inference_config = (
-        initialize_tabpfn_model(
-            model_path=calling_instance.model_path,  # pyright: ignore[reportArgumentType]
-            which=model_type,
-            fit_mode=calling_instance.fit_mode,  # pyright: ignore[reportArgumentType]
+    if model_type == "regressor":
+        (
+            calling_instance.model_,
+            calling_instance.config_,
+            calling_instance.bardist_,
+        ) = initialize_tabpfn_model(
+            model_path=calling_instance.model_path,
+            which="regressor",
+            fit_mode=calling_instance.fit_mode,  # Use the instance's fit_mode
         )
-    )
-    calling_instance.models_ = models
-    calling_instance.configs_ = architecture_configs
-    if model_type == "regressor" and maybe_bardist is not None:
-        calling_instance.znorm_space_bardist_ = maybe_bardist
+    elif model_type == "classifier":
+        (calling_instance.model_, calling_instance.config_, _) = (
+            initialize_tabpfn_model(
+                model_path=calling_instance.model_path,
+                which="classifier",
+                fit_mode=calling_instance.fit_mode,  # Use the instance's fit_mode
+            )
+        )
+    else:
+        raise ValueError(f"Invalid model_type: {model_type}")
 
-    calling_instance.devices_ = infer_devices(calling_instance.device)
+    calling_instance.device_ = infer_device_and_type(calling_instance.device)
     (
         calling_instance.use_autocast_,
         calling_instance.forced_inference_dtype_,
         byte_size,
     ) = determine_precision(
-        calling_instance.inference_precision, calling_instance.devices_
+        calling_instance.inference_precision, calling_instance.device_
     )
+    calling_instance.model_.to(calling_instance.device_)
 
-    inference_config = inference_config.override_with_user_input(
-        user_config=calling_instance.inference_config
-    )
+    # Build the interface_config
+    _config = ModelInterfaceConfig.from_user_input(
+        inference_config=calling_instance.inference_config,
+    )  # shorter alias
 
-    calling_instance.inference_config_ = inference_config
+    calling_instance.interface_config_ = _config
 
-    outlier_removal_std = inference_config.OUTLIER_REMOVAL_STD
+    outlier_removal_std = _config.OUTLIER_REMOVAL_STD
     if outlier_removal_std == "auto":
         default_stds = {
-            "regressor": inference_config._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD,
-            "classifier": inference_config._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD,
+            "regressor": _config._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD,
+            "classifier": _config._CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD,
         }
         try:
             outlier_removal_std = default_stds[model_type]
@@ -561,9 +497,10 @@ def initialize_model_variables_helper(
             raise ValueError(f"Invalid model_type: {model_type}") from e
 
     update_encoder_params(  # Use the renamed function if available, or original one
-        models=calling_instance.models_,
+        model=calling_instance.model_,
         remove_outliers_std=outlier_removal_std,
         seed=static_seed,
+        inplace=True,
         differentiable_input=calling_instance.differentiable_input,
     )
     return byte_size, rng
diff --git a/src/tabpfn/classifier.py b/src/tabpfn/classifier.py
index b8171d1..9f64d3e 100644
--- a/src/tabpfn/classifier.py
+++ b/src/tabpfn/classifier.py
@@ -18,67 +18,48 @@
 
 from __future__ import annotations
 
-import copy
 import logging
-import warnings
-from collections.abc import Callable, Sequence
+import typing
+from collections.abc import Sequence
 from pathlib import Path
-from typing import TYPE_CHECKING, Annotated, Any, Literal
-from typing_extensions import Self, deprecated
+from typing import TYPE_CHECKING, Any, Literal
+from typing_extensions import Self
 
 import numpy as np
 import torch
 from sklearn import config_context
 from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
 from sklearn.preprocessing import LabelEncoder
-from tabpfn_common_utils.telemetry import track_model_call
-from tabpfn_common_utils.telemetry.interactive import ping
 
 from tabpfn.base import (
-    ClassifierModelSpecs,
+    _initialize_model_variables_helper,
     check_cpu_warning,
     create_inference_engine,
     determine_precision,
     get_preprocessed_datasets_helper,
-    initialize_model_variables_helper,
 )
 from tabpfn.constants import (
     PROBABILITY_EPSILON_ROUND_ZERO,
     SKLEARN_16_DECIMAL_PRECISION,
-    ModelVersion,
     XType,
     YType,
 )
 from tabpfn.inference import InferenceEngine, InferenceEngineBatchedNoPreprocessing
-from tabpfn.inference_tuning import (
-    ClassifierEvalMetrics,
-    ClassifierTuningConfig,
-    find_optimal_classification_thresholds,
-    find_optimal_temperature,
-    get_tuning_splits,
-    resolve_tuning_config,
-)
-from tabpfn.model_loading import (
-    ModelSource,
-    get_cache_dir,
-    load_fitted_tabpfn_model,
-    save_fitted_tabpfn_model,
-)
+from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
 from tabpfn.preprocessing import (
     ClassifierEnsembleConfig,
     DatasetCollectionWithPreprocessing,
     EnsembleConfig,
     PreprocessorConfig,
+    default_classifier_preprocessor_configs,
 )
-from tabpfn.preprocessors.preprocessing_helpers import get_ordinal_encoder
 from tabpfn.utils import (
-    DevicesSpecification,
-    balance_probas_by_class_counts,
-    fix_dtypes,
-    get_embeddings,
+    _fix_dtypes,
+    _get_embeddings,
+    _get_ordinal_encoder,
+    _process_text_na_dataframe,
     infer_categorical_features,
     infer_random_state,
-    process_text_na_dataframe,
     validate_X_predict,
     validate_Xy_fit,
 )
@@ -88,44 +69,30 @@ if TYPE_CHECKING:
     from sklearn.compose import ColumnTransformer
     from torch.types import _dtype
 
-    from tabpfn.architectures.base.memory import MemorySavingMode
-    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
-    from tabpfn.inference_config import InferenceConfig
+    from tabpfn.architectures.interface import ArchitectureConfig
+    from tabpfn.config import ModelInterfaceConfig
 
     try:
         from sklearn.base import Tags
     except ImportError:
         Tags = Any
 
-DEFAULT_CLASSIFICATION_EVAL_METRIC = ClassifierEvalMetrics.ACCURACY
-
 
 class TabPFNClassifier(ClassifierMixin, BaseEstimator):
     """TabPFNClassifier class."""
 
-    configs_: list[ArchitectureConfig]
-    """The configurations of the loaded models to be used for inference.
-
-    The concrete type of these configs is defined by the architectures in use and should
-    be inspected at runtime, but they will be subclasses of ArchitectureConfig.
-    """
-
-    models_: list[Architecture]
-    """The loaded models to be used for inference.
+    config_: ArchitectureConfig
+    """The configuration of the loaded model to be used for inference.
 
-    The models can be different PyTorch modules, but will be subclasses of Architecture.
+    The concrete type of this config is defined by the arhitecture in use and should be
+    inspected at runtime, but it will be a subclass of ArchitectureConfig.
     """
 
-    inference_config_: InferenceConfig
+    interface_config_: ModelInterfaceConfig
     """Additional configuration of the interface for expert users."""
 
-    devices_: tuple[torch.device, ...]
-    """The devices determined to be used.
-
-    The devices are determined based on the `device` argument to the constructor, and
-    the devices available on the system. If multiple devices are listed, currently only
-    the first is used for inference.
-    """
+    device_: torch.device
+    """The device determined to be used."""
 
     feature_names_in_: npt.NDArray[Any]
     """The feature names of the input data.
@@ -170,17 +137,6 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
     preprocessor_: ColumnTransformer
     """The column transformer used to preprocess the input data to be numeric."""
 
-    tuned_classification_thresholds_: npt.NDArray[Any] | None
-    """The tuned classification thresholds for each class or None if no tuning is
-    specified."""
-
-    eval_metric_: ClassifierEvalMetrics
-    """The validated evaluation metric to optimize for during prediction."""
-
-    softmax_temperature_: float
-    """The softmax temperature used for prediction. This is set to the default softmax
-    temperature if no temperature tuning is done"""
-
     def __init__(  # noqa: PLR0913
         self,
         *,
@@ -189,14 +145,8 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         softmax_temperature: float = 0.9,
         balance_probabilities: bool = False,
         average_before_softmax: bool = False,
-        model_path: str
-        | Path
-        | list[str]
-        | list[Path]
-        | Literal["auto"]
-        | ClassifierModelSpecs
-        | list[ClassifierModelSpecs] = "auto",
-        device: DevicesSpecification = "auto",
+        model_path: str | Path | Literal["auto"] = "auto",
+        device: str | torch.device | Literal["auto"] = "auto",
         ignore_pretraining_limits: bool = False,
         inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
         fit_mode: Literal[
@@ -205,20 +155,13 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             "fit_with_cache",
             "batched",
         ] = "fit_preprocessors",
-        memory_saving_mode: MemorySavingMode = "auto",
+        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
         random_state: int | np.random.RandomState | np.random.Generator | None = 0,
-        n_jobs: Annotated[int | None, deprecated("Use n_preprocessing_jobs")] = None,
-        n_preprocessing_jobs: int = 1,
-        inference_config: dict | InferenceConfig | None = None,
+        n_jobs: int = -1,
+        inference_config: dict | ModelInterfaceConfig | None = None,
         differentiable_input: bool = False,
-        eval_metric: str | ClassifierEvalMetrics | None = None,
-        tuning_config: dict | ClassifierTuningConfig | None = None,
     ) -> None:
-        """Construct a TabPFN classifier.
-
-        This constructs a classifier using the latest model and settings. If you would
-        like to use a previous model version, use `create_default_for_version()`
-        instead. You can also use `model_path` to specify a particular model.
+        """A TabPFN interface for classification.
 
         Args:
             n_estimators:
@@ -244,9 +187,7 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
                 The temperature for the softmax function. This is used to control the
                 confidence of the model's predictions. Lower values make the model's
                 predictions more confident. This is only applied when predicting during
-                a post-processing step. Set `softmax_temperature=1.0` for no effect. Be
-                advised that `.predict()` does not currently sample, so this setting is
-                only relevant for `.predict_proba()` and `.predict_logits()`.
+                a post-processing step. Set `softmax_temperature=1.0` for no effect.
 
             balance_probabilities:
                 Whether to balance the probabilities based on the class distribution
@@ -271,8 +212,6 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
 
             model_path:
                 The path to the TabPFN model file, i.e., the pre-trained weights.
-                Can be a list of paths to load multiple models. If a list is provided,
-                the models are applied across different estimators.
 
                 - If `"auto"`, the model will be downloaded upon first use. This
                   defaults to your system cache directory, but can be overwritten
@@ -282,18 +221,14 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
                   downloaded to this location.
 
             device:
-                The device(s) to use for inference.
-
-                If "auto": a single device is selected based on availability in the
-                following order of priority: "cuda:0", "mps", "cpu".
+                The device to use for inference with TabPFN. If set to "auto", the
+                device is selected based on availability in the following order of
+                priority: "cuda", "mps", and then "cpu". You can also set the device
+                manually to one of these options.
 
-                To manually select a single device: specify a PyTorch device string e.g.
-                "cuda:1". See PyTorch's documentation for information about supported
-                devices.
+                See PyTorch's documentation on devices for more information about
+                supported devices.
 
-                To use several GPUs: specify a list of PyTorch GPU device strings, e.g.
-                ["cuda:0", "cuda:1"]. This can dramatically speed up inference for
-                larger datasets, by executing the estimators in parallel on the GPUs.
 
             ignore_pretraining_limits:
                 Whether to ignore the pre-training limits of the model. The TabPFN
@@ -310,12 +245,10 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
 
                 !!! note
 
-                    For version 2.5, the pre-training limits are:
+                    The current pre-training limits are:
 
-                    - 50_000 samples/rows
-                    - 2_000 features/columns (Note that for more than 500 features we
-                        subsample 500 features per estimator. It is therefore important
-                        to use a sufficiently large number of `n_estimators`.)
+                    - 10_000 samples/rows
+                    - 500 features/columns
                     - 10 classes, this is not ignorable and will raise an error
                       if the model is used with more classes.
 
@@ -363,24 +296,32 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
                   class in Fine-Tuning. The fit_from_preprocessed() function sets this
                   attribute internally.
 
-            memory_saving_mode:
-                Enable GPU/CPU memory saving mode. This can both avoid out-of-memory
-                errors and improve fit+predict speed by reducing memory pressure.
-
-                It saves memory by automatically batching certain model computations
-                within TabPFN.
-
-                - If "auto": memory saving mode is enabled/disabled automatically based
-                    on a heuristic
-                - If True/False: memory saving mode is forced enabled/disabled.
 
-                If speed is important to your application, you may wish to manually tune
-                this option by comparing the time taken for fit+predict with it set to
-                False and True.
+            memory_saving_mode:
+                Enable GPU/CPU memory saving mode. This can help to prevent
+                out-of-memory errors that result from computations that would consume
+                more memory than available on the current device. We save memory by
+                automatically batching certain model computations within TabPFN to
+                reduce the total required memory. The options are:
+
+                - If `bool`, enable/disable memory saving mode.
+                - If `"auto"`, we will estimate the amount of memory required for the
+                  forward pass and apply memory saving if it is more than the
+                  available GPU/CPU memory. This is the recommended setting as it
+                  allows for speed-ups and prevents memory errors depending on
+                  the input data.
+                - If `float` or `int`, we treat this value as the maximum amount of
+                  available GPU/CPU memory (in GB). We will estimate the amount
+                  of memory required for the forward pass and apply memory saving
+                  if it is more than this value. Passing a float or int value for
+                  this parameter is the same as setting it to True and explicitly
+                  specifying the maximum free available memory.
 
                 !!! warning
                     This does not batch the original input data. We still recommend to
-                    batch the test set as necessary if you run out of memory.
+                    batch this as necessary if you run into memory errors! For example,
+                    if the entire input data does not fit into memory, even the memory
+                    save mode will not prevent memory errors.
 
             random_state:
                 Controls the randomness of the model. Pass an int for reproducible
@@ -401,50 +342,28 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
                     passing `USE_SKLEARN_16_DECIMAL_PRECISION=True` as kwarg.
 
             n_jobs:
-                Deprecated, use `n_preprocessing_jobs` instead.
-                This parameter never had any effect.
-
-            n_preprocessing_jobs:
-                The number of worker processes to use for the preprocessing.
-
-                If `1`, the preprocessing will be performed in the current process,
-                parallelised across multiple CPU cores. If `>1` and `n_estimators > 1`,
-                then different estimators will be dispatched to different processes.
+                The number of workers for tasks that can be parallelized across CPU
+                cores. Currently, this is used for preprocessing the data in parallel
+                (if `n_estimators > 1`).
 
-                We strongly recommend setting this to 1, which has the lowest overhead
-                and can often fully utilise the CPU. Values >1 can help if you have lots
-                of CPU cores available, but can also be slower.
+                - If `-1`, all available CPU cores are used.
+                - If `int`, the number of CPU cores to use is determined by `n_jobs`.
 
             inference_config:
                 For advanced users, additional advanced arguments that adjust the
                 behavior of the model interface.
-                See [tabpfn.inference_config.InferenceConfig][] for details and options.
+                See [tabpfn.constants.ModelInterfaceConfig][] for details and options.
 
-                - If `None`, the default InferenceConfig is used.
+                - If `None`, the default ModelInterfaceConfig is used.
                 - If `dict`, the key-value pairs are used to update the default
-                  `InferenceConfig`. Raises an error if an unknown key is passed.
-                - If `InferenceConfig`, the object is used as the configuration.
+                  `ModelInterfaceConfig`. Raises an error if an unknown key is passed.
+                - If `ModelInterfaceConfig`, the object is used as the configuration.
 
             differentiable_input:
                 If true, the preprocessing will be adapted to be end-to-end
                 differentiable with PyTorch.
                 This is useful for explainability and prompt-tuning, essential
                 in the prompttuning code.
-
-            eval_metric:
-                Metric by which predictions will be ultimately evaluated on test data.
-                This can be used to improve this metric on validation data by
-                calibrating the model's probabilities and tuning the decision
-                thresholds during the `fit()/predict()` calls. The tuning can be
-                enabled by configuring the `tuning_config` argument, see below.
-                For currently supported metrics, see
-                [tabpfn.classifier.ClassifierEvalMetrics][].
-
-            tuning_config:
-                The settings to use to tune the model's predictions for the specified
-                `eval_metric`. See
-                [tabpfn.inference_tuning.ClassifierTuningConfig][] for details
-                and options.
         """
         super().__init__()
         self.n_estimators = n_estimators
@@ -458,79 +377,17 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
             inference_precision
         )
-        self.fit_mode = fit_mode
+        self.fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = (
+            fit_mode
+        )
         self.memory_saving_mode: bool | Literal["auto"] | float | int = (
             memory_saving_mode
         )
         self.random_state = random_state
+        self.n_jobs = n_jobs
         self.inference_config = inference_config
         self.differentiable_input = differentiable_input
 
-        if n_jobs is not None:
-            warnings.warn(
-                "TabPFNClassifier(n_jobs=...) is deprecated and has no effect. "
-                "Use `n_preprocessing_jobs` instead.",
-                DeprecationWarning,
-                stacklevel=2,
-            )
-        self.n_jobs = n_jobs
-        self.n_preprocessing_jobs = n_preprocessing_jobs
-        self.eval_metric = eval_metric
-        self.tuning_config = tuning_config
-
-        # Ping the usage service if telemetry enabled
-        ping()
-
-    @classmethod
-    def create_default_for_version(cls, version: ModelVersion, **overrides) -> Self:
-        """Construct a classifier that uses the given version of the model.
-
-        In addition to selecting the model, this also configures certain settings to the
-        default values associated with this model version.
-
-        Any kwargs will override the default settings.
-        """
-        if version == ModelVersion.V2:
-            options = {
-                "model_path": str(
-                    get_cache_dir() / ModelSource.get_classifier_v2().default_filename
-                ),
-                "n_estimators": 8,
-                "softmax_temperature": 0.9,
-            }
-        elif version == ModelVersion.V2_5:
-            options = {
-                "model_path": str(
-                    get_cache_dir() / ModelSource.get_classifier_v2_5().default_filename
-                ),
-                "n_estimators": 8,
-                "softmax_temperature": 0.9,
-            }
-        else:
-            raise ValueError(f"Unknown version: {version}")
-
-        options.update(overrides)
-
-        return cls(**options)
-
-    @property
-    def model_(self) -> Architecture:
-        """The model used for inference.
-
-        This is set after the model is loaded and initialized.
-        """
-        if not hasattr(self, "models_"):
-            raise ValueError(
-                "The model has not been initialized yet. Please initialize the model "
-                "before using the `model_` property."
-            )
-        if len(self.models_) > 1:
-            raise ValueError(
-                "The `model_` property is not supported when multiple models are used. "
-                "Use `models_` instead."
-            )
-        return self.models_[0]
-
     # TODO: We can remove this from scikit-learn lower bound of 1.6
     def _more_tags(self) -> dict[str, Any]:
         return {
@@ -548,10 +405,8 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         self,
         X_raw: XType | list[XType],
         y_raw: YType | list[YType],
-        split_fn: Callable,
+        split_fn,
         max_data_size: None | int = 10000,
-        *,
-        equal_split_size: bool = True,
     ) -> DatasetCollectionWithPreprocessing:
         """Transforms raw input data into a collection of datasets,
         with varying preprocessings.
@@ -570,11 +425,6 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             split_fn: A function to dissect a dataset into train and test partition.
             max_data_size: Maximum allowed number of samples in one dataset.
             If None, datasets are not splitted.
-            equal_split_size: If True, splits data into equally sized chunks under
-            max_data_size.
-            If False, splits into chunks of size `max_data_size`, with
-            the last chunk having the remainder samples but is dropped if its
-            size is less than 2.
         """
         return get_preprocessed_datasets_helper(
             self,
@@ -583,20 +433,16 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             split_fn,
             max_data_size,
             model_type="classifier",
-            equal_split_size=equal_split_size,
         )
 
     def _initialize_model_variables(self) -> tuple[int, np.random.Generator]:
         """Perform initialization of the model, return determined byte_size
         and RNG object.
         """
-        return initialize_model_variables_helper(self, "classifier")
+        return _initialize_model_variables_helper(self, "classifier")
 
     def _initialize_dataset_preprocessing(
-        self,
-        X: XType,
-        y: YType,
-        rng,  # noqa: ANN001
+        self, X: XType, y: YType, rng
     ) -> tuple[list[ClassifierEnsembleConfig], XType, YType]:
         """Internal preprocessing method for input arguments.
         Returns ClassifierEnsembleConfigs, inferred categorical indices,
@@ -608,13 +454,13 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             y,
             estimator=self,
             ensure_y_numeric=False,
-            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
-            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
+            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
+            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
             ignore_pretraining_limits=self.ignore_pretraining_limits,
         )
 
         check_cpu_warning(
-            self.devices_, X, allow_cpu_override=self.ignore_pretraining_limits
+            self.device, X, allow_cpu_override=self.ignore_pretraining_limits
         )
 
         if feature_names_in is not None:
@@ -640,7 +486,7 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             self.classes_ = torch.arange(self.n_classes_)
 
         # TODO: Support more classes with a fallback strategy.
-        if self.n_classes_ > self.inference_config_.MAX_NUMBER_OF_CLASSES:
+        if self.n_classes_ > self.interface_config_.MAX_NUMBER_OF_CLASSES:
             raise ValueError(
                 f"Number of classes {self.n_classes_} exceeds the maximal number of "
                 "classes supported by TabPFN. Consider using a strategy to reduce "
@@ -656,19 +502,19 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             self.inferred_categorical_indices_ = infer_categorical_features(
                 X=X,
                 provided=self.categorical_features_indices,
-                min_samples_for_inference=self.inference_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
-                max_unique_for_category=self.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
-                min_unique_for_numerical=self.inference_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
+                min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
+                max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
+                min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
             )
-            preprocessor_configs = self.inference_config_.PREPROCESS_TRANSFORMS
+            preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS
 
             # Will convert inferred categorical indices to category dtype,
             # to be picked up by the ord_encoder, as well
             # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
-            X = fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
+            X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
             # Ensure categories are ordinally encoded
-            ord_encoder = get_ordinal_encoder()
-            X = process_text_na_dataframe(X, ord_encoder=ord_encoder, fit_encoder=True)
+            ord_encoder = _get_ordinal_encoder()
+            X = _process_text_na_dataframe(X, ord_encoder=ord_encoder, fit_encoder=True)
 
             assert isinstance(X, np.ndarray)
             self.preprocessor_ = ord_encoder
@@ -676,27 +522,30 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         else:  # Minimal preprocessing for prompt tuning
             self.inferred_categorical_indices_ = []
             self.preprocessor_ = None
-            preprocessor_configs = [PreprocessorConfig("none", differentiable=True)]
+            preprocess_transforms = [PreprocessorConfig("none", differentiable=True)]
 
         ensemble_configs = EnsembleConfig.generate_for_classification(
-            num_estimators=self.n_estimators,
-            subsample_size=self.inference_config_.SUBSAMPLE_SAMPLES,
-            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
-            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
-            polynomial_features=self.inference_config_.POLYNOMIAL_FEATURES,
+            n=self.n_estimators,
+            subsample_size=self.interface_config_.SUBSAMPLE_SAMPLES,
+            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
+            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
+            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
             max_index=len(X),
-            preprocessor_configs=preprocessor_configs,
-            class_shift_method=self.inference_config_.CLASS_SHIFT_METHOD
+            preprocessor_configs=typing.cast(
+                "Sequence[PreprocessorConfig]",
+                preprocess_transforms
+                if preprocess_transforms is not None
+                else default_classifier_preprocessor_configs(),
+            ),
+            class_shift_method=self.interface_config_.CLASS_SHIFT_METHOD
             if not self.differentiable_input
             else None,
             n_classes=self.n_classes_,
             random_state=rng,
-            num_models=len(self.models_),
         )
         assert len(ensemble_configs) == self.n_estimators
         return ensemble_configs, X, y
 
-    @track_model_call("fit", param_names=["X_preprocessed", "y_preprocessed"])
     def fit_from_preprocessed(
         self,
         X_preprocessed: list[torch.Tensor],
@@ -704,7 +553,7 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         cat_ix: list[list[int]],
         configs: list[list[EnsembleConfig]],
         *,
-        no_refit: bool = True,
+        no_refit=True,
     ) -> TabPFNClassifier:
         """Used in Fine-Tuning. Fit the model to preprocessed inputs from torch
         dataloader inside a training loop a Dataset provided by
@@ -730,11 +579,11 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             self.fit_mode = "batched"
 
         # If there is a model, and we are lazy, we skip reinitialization
-        if not hasattr(self, "models_") or not no_refit:
+        if not hasattr(self, "model_") or not no_refit:
             byte_size, rng = self._initialize_model_variables()
         else:
             _, _, byte_size = determine_precision(
-                self.inference_precision, self.devices_
+                self.inference_precision, self.device_
             )
             rng = None
 
@@ -742,13 +591,13 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         self.executor_ = create_inference_engine(
             X_train=X_preprocessed,
             y_train=y_preprocessed,
-            models=self.models_,
+            model=self.model_,
             ensemble_configs=configs,
             cat_ix=cat_ix,
             fit_mode="batched",
-            devices_=self.devices_,
+            device_=self.device_,
             rng=rng,
-            n_preprocessing_jobs=self.n_preprocessing_jobs,
+            n_jobs=self.n_jobs,
             byte_size=byte_size,
             forced_inference_dtype_=self.forced_inference_dtype_,
             memory_saving_mode=self.memory_saving_mode,
@@ -758,52 +607,14 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
 
         return self
 
-    def _get_tuning_classifier(self, **overwrite_kwargs: Any) -> TabPFNClassifier:
-        """Return a fresh classifier configured for holdout tuning."""
-        params = self.get_params(deep=False)
-
-        # Avoids sharing mutable config across instances
-        for key in params:
-            try:
-                if isinstance(params.get(key), dict):
-                    params[key] = copy.deepcopy(params[key])
-            except Exception as e:  # noqa: BLE001
-                logging.warning(
-                    "Error during initialization of tuning classifier when trying "
-                    f"to deepcopy configuration with name `{key}`: {e}. "
-                    "Falling back to original configuration"
-                )
-
-        forced = {
-            "fit_mode": "fit_preprocessors",
-            "differentiable_input": False,
-            "tuning_config": None,  # never tune inside tuning
-        }
-
-        params.update(forced)
-        params.update(overwrite_kwargs)
-
-        return TabPFNClassifier(**params)
-
     @config_context(transform_output="default")  # type: ignore
-    @track_model_call(model_method="fit", param_names=["X", "y"])
-    def fit(
-        self,
-        X: XType,
-        y: YType,
-    ) -> Self:
+    def fit(self, X: XType, y: YType) -> Self:
         """Fit the model.
 
         Args:
             X: The input data.
             y: The target variable.
-
-        Returns:
-            self
         """
-        # Validate eval_metric here instead of in __init__ as per sklearn convention
-        self.eval_metric_ = _validate_eval_metric(self.eval_metric)
-
         if self.fit_mode == "batched":
             logging.warning(
                 "The model was in 'batched' mode, likely after finetuning. "
@@ -812,30 +623,26 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             )
             self.fit_mode = "fit_preprocessors"
 
-        if not hasattr(self, "models_") or not self.differentiable_input:
+        if not hasattr(self, "model_") or not self.differentiable_input:
             byte_size, rng = self._initialize_model_variables()
             ensemble_configs, X, y = self._initialize_dataset_preprocessing(X, y, rng)
         else:  # already fitted and prompt_tuning mode: no cat. features
             _, rng = infer_random_state(self.random_state)
             _, _, byte_size = determine_precision(
-                self.inference_precision, self.devices_
+                self.inference_precision, self.device_
             )
 
-        self._maybe_calibrate_temperature_and_tune_decision_thresholds(
-            X=X,
-            y=y,
-        )
-
+        # Create the inference engine
         self.executor_ = create_inference_engine(
             X_train=X,
             y_train=y,
-            models=self.models_,
+            model=self.model_,
             ensemble_configs=ensemble_configs,
             cat_ix=self.inferred_categorical_indices_,
             fit_mode=self.fit_mode,
-            devices_=self.devices_,
+            device_=self.device_,
             rng=rng,
-            n_preprocessing_jobs=self.n_preprocessing_jobs,
+            n_jobs=self.n_jobs,
             byte_size=byte_size,
             forced_inference_dtype_=self.forced_inference_dtype_,
             memory_saving_mode=self.memory_saving_mode,
@@ -845,145 +652,7 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
 
         return self
 
-    def _maybe_calibrate_temperature_and_tune_decision_thresholds(
-        self,
-        X: XType,
-        y: YType,
-    ) -> None:
-        """If this class was initialized with a 'tuning_config', calibrate and tune.
-
-        This first computes scores on validation holdout data and then calibrates the
-        softmax temperature and tunes the decision thresholds as per the tuning
-        configuration. Results are stored in the 'tuned_classification_thresholds_' and
-        'softmax_temperature_' attributes.
-        """
-        assert self.eval_metric_ is not None
-
-        # Always set this to stay compatible with sklearn interface.
-        self.tuned_classification_thresholds_ = None
-        self.softmax_temperature_ = self.softmax_temperature
-
-        tuning_config_resolved = resolve_tuning_config(
-            tuning_config=self.tuning_config,
-            num_samples=X.shape[0],
-        )
-        if tuning_config_resolved is None:
-            if self.eval_metric_ is ClassifierEvalMetrics.F1:
-                warnings.warn(
-                    f"You specified '{self.eval_metric_}' as the eval metric but "
-                    "haven't specified any tuning configuration. Consider configuring "
-                    "tuning via the `tuning_config` argument of the TabPFNClassifier "
-                    "to improve predictive performance.",
-                    UserWarning,
-                    stacklevel=2,
-                )
-            if self.eval_metric_ is ClassifierEvalMetrics.BALANCED_ACCURACY:
-                warnings.warn(
-                    f"You specified '{self.eval_metric_}' as the eval metric but "
-                    "haven't specified any tuning configuration. "
-                    f"For metric '{self.eval_metric_}' we recommend "
-                    "balancing the probabilities by class counts which can be achieved "
-                    "by setting `balance_probabilities` to True.",
-                    UserWarning,
-                    stacklevel=2,
-                )
-            return
-
-        if self.eval_metric_ is ClassifierEvalMetrics.ROC_AUC:
-            warnings.warn(
-                f"You specified '{self.eval_metric_}' as the eval metric with "
-                "threshold tuning or temperature calibration enabled. "
-                "ROC AUC is independent of these tunings and they will not "
-                "improve this metric. Consider disabling them.",
-                UserWarning,
-                stacklevel=2,
-            )
-
-        holdout_raw_logits, holdout_y_true = self._compute_holdout_validation_data(
-            X=X,
-            y=y,
-            holdout_frac=float(tuning_config_resolved.tuning_holdout_frac),
-            n_folds=int(tuning_config_resolved.tuning_n_folds),
-        )
-
-        # WARNING: ensure the calibration happens before threshold tuning!
-        if tuning_config_resolved.calibrate_temperature:
-            calibrated_softmax_temperature = self._get_calibrated_softmax_temperature(
-                holdout_raw_logits=holdout_raw_logits,
-                holdout_y_true=holdout_y_true,
-            )
-            self.softmax_temperature_ = calibrated_softmax_temperature
-
-        if tuning_config_resolved.tune_decision_thresholds:
-            holdout_probas = (
-                self.logits_to_probabilities(holdout_raw_logits)
-                .float()
-                .detach()
-                .cpu()
-                .numpy()
-            )
-            tuned_classification_thresholds = find_optimal_classification_thresholds(
-                metric_name=self.eval_metric_,
-                y_true=holdout_y_true,
-                y_pred_probas=holdout_probas,
-                n_classes=self.n_classes_,
-            )
-            self.tuned_classification_thresholds_ = tuned_classification_thresholds
-
-    def _compute_holdout_validation_data(
-        self,
-        X: XType,
-        y: YType,
-        holdout_frac: float,
-        n_folds: int,
-    ) -> tuple[np.ndarray, np.ndarray]:
-        """Compute holdout validation data.
-
-        Returns:
-            tuple[np.ndarray, np.ndarray]:
-                - holdout_raw_logits: Array of holdout raw logits
-                    (shape `[n_estimators, n_holdout_samples, n_classes]`).
-                - holdout_y_true: Array of holdout y true labels
-                    (shape `[n_holdout_samples]`).
-        """
-        splits = get_tuning_splits(
-            X=copy.deepcopy(X),
-            y=copy.deepcopy(y),
-            holdout_frac=holdout_frac,
-            random_state=self.random_state,
-            n_splits=n_folds,
-        )
-
-        holdout_raw_logits = []
-        holdout_y_true = []
-        # suffixes: Nt=num_train_samples, F=num_features, Nh=num_holdout_samples
-        for X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh in splits:
-            holdout_y_true.append(y_holdout_Nh)
-            calibration_classifier = self._get_tuning_classifier()
-            with warnings.catch_warnings():
-                # Filter expected warnings during tuning
-                warnings.filterwarnings(
-                    "ignore",
-                    message=".*haven't specified any tuning configuration*",
-                    category=UserWarning,
-                )
-                calibration_classifier.fit(X_train_NtF, y_train_Nt)
-
-            # E=num estimators, Nh=num holdout samples, C=num classes
-            raw_logits_ENhC = calibration_classifier.predict_raw_logits(X=X_holdout_NhF)
-            holdout_raw_logits.append(raw_logits_ENhC)
-
-        holdout_raw_logits_all = np.concatenate(holdout_raw_logits, axis=1)
-        holdout_y_true__all = np.concatenate(holdout_y_true, axis=0)
-        return holdout_raw_logits_all, holdout_y_true__all
-
-    def _raw_predict(
-        self,
-        X: XType,
-        *,
-        return_logits: bool,
-        return_raw_logits: bool = False,
-    ) -> torch.Tensor:
+    def _raw_predict(self, X: XType, *, return_logits: bool) -> torch.Tensor:
         """Internal method to run prediction.
 
         Handles input validation, preprocessing, and the forward pass.
@@ -992,31 +661,23 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
 
         Args:
             X: The input data for prediction.
-            return_logits: If True, the logits are returned. Otherwise,
+            return_logits: If True, the raw logits are returned. Otherwise,
                            probabilities are returned after softmax and other
                            post-processing steps.
-            return_raw_logits: If True, returns the raw logits without
-                averaging estimators or temperature scaling.
 
         Returns:
             The raw torch.Tensor output, either logits or probabilities,
-            depending on `return_logits` and `return_raw_logits`.
+            depending on `return_logits`.
         """
         check_is_fitted(self)
 
         if not self.differentiable_input:
             X = validate_X_predict(X, self)
-            X = fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
-            X = process_text_na_dataframe(X, ord_encoder=self.preprocessor_)
+            X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
+            X = _process_text_na_dataframe(X, ord_encoder=self.preprocessor_)
 
-        return self.forward(
-            X,
-            use_inference_mode=True,
-            return_logits=return_logits,
-            return_raw_logits=return_raw_logits,
-        )
+        return self.forward(X, use_inference_mode=True, return_logits=return_logits)
 
-    @track_model_call(model_method="predict", param_names=["X"])
     def predict(self, X: XType) -> np.ndarray:
         """Predict the class labels for the provided input samples.
 
@@ -1026,15 +687,14 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         Returns:
             The predicted class labels as a NumPy array.
         """
-        probas = self._predict_proba(X=X)
-        y_pred = np.argmax(probas, axis=1)
+        proba = self.predict_proba(X)
+        y_pred = np.argmax(proba, axis=1)
         if hasattr(self, "label_encoder_") and self.label_encoder_ is not None:
             return self.label_encoder_.inverse_transform(y_pred)
 
         return y_pred
 
     @config_context(transform_output="default")
-    @track_model_call(model_method="predict", param_names=["X"])
     def predict_logits(self, X: XType) -> np.ndarray:
         """Predict the raw logits for the provided input samples.
 
@@ -1050,36 +710,10 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         logits_tensor = self._raw_predict(X, return_logits=True)
         return logits_tensor.float().detach().cpu().numpy()
 
-    @config_context(transform_output="default")
-    @track_model_call(model_method="predict", param_names=["X"])
-    def predict_raw_logits(self, X: XType) -> np.ndarray:
-        """Predict the raw logits for the provided input samples.
-
-        Logits represent the unnormalized log-probabilities of the classes
-        before the softmax activation function is applied. In contrast to the
-        `predict_logits` method, this method returns the raw logits for each
-        estimator, without averaging estimators or temperature scaling.
-
-        Args:
-            X: The input data for prediction.
-
-        Returns:
-            An array of predicted logits for each estimator,
-            Shape (n_estimators, n_samples, n_classes).
-        """
-        logits_tensor = self._raw_predict(
-            X,
-            return_logits=False,
-            return_raw_logits=True,
-        )
-        return logits_tensor.float().detach().cpu().numpy()
-
-    @track_model_call(model_method="predict", param_names=["X"])
+    @config_context(transform_output="default")  # type: ignore
     def predict_proba(self, X: XType) -> np.ndarray:
         """Predict the probabilities of the classes for the provided input samples.
 
-        This is a wrapper around the `_predict_proba` method.
-
         Args:
             X: The input data for prediction.
 
@@ -1087,87 +721,21 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             The predicted probabilities of the classes as a NumPy array.
             Shape (n_samples, n_classes).
         """
-        return self._predict_proba(X)
+        proba_tensor = self._raw_predict(X, return_logits=False)
+        output = proba_tensor.float().detach().cpu().numpy()
 
-    @config_context(transform_output="default")  # type: ignore
-    def _predict_proba(self, X: XType) -> np.ndarray:
-        """Predict the probabilities of the classes for the provided input samples.
-
-        Args:
-            X: The input data for prediction.
-
-        Returns:
-            The predicted probabilities of the classes as a NumPy array.
-            Shape (n_samples, n_classes).
-        """
-        probas = (
-            self._raw_predict(X, return_logits=False).float().detach().cpu().numpy()
-        )
-        probas = self._maybe_reweight_probas(probas=probas)
-        if self.inference_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
-            probas = np.around(probas, decimals=SKLEARN_16_DECIMAL_PRECISION)
-            probas = np.where(probas < PROBABILITY_EPSILON_ROUND_ZERO, 0.0, probas)
+        if self.interface_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
+            output = np.around(output, decimals=SKLEARN_16_DECIMAL_PRECISION)
+            output = np.where(output < PROBABILITY_EPSILON_ROUND_ZERO, 0.0, output)
 
         # Ensure probabilities sum to 1 in case of minor floating point inaccuracies
         # going from torch to numpy
-        return probas / probas.sum(axis=1, keepdims=True)  # type: ignore
-
-    def _get_calibrated_softmax_temperature(
-        self,
-        holdout_raw_logits: np.ndarray,
-        holdout_y_true: np.ndarray,
-    ) -> float:
-        """Calibrate temperature based on the holdout logits and true labels."""
-
-        def logits_to_probabilities_fn(
-            raw_logits: np.ndarray | torch.Tensor,
-            softmax_temperature: float,
-        ) -> np.ndarray:
-            return (
-                self.logits_to_probabilities(
-                    raw_logits=raw_logits,
-                    softmax_temperature=softmax_temperature,
-                    average_before_softmax=self.average_before_softmax,
-                    balance_probabilities=self.balance_probabilities,
-                )
-                .float()
-                .detach()
-                .cpu()
-                .numpy()
-            )
-
-        return find_optimal_temperature(
-            raw_logits=holdout_raw_logits,
-            y_true=holdout_y_true,
-            logits_to_probabilities_fn=logits_to_probabilities_fn,
-            current_default_temperature=self.softmax_temperature_,
-        )
-
-    def _maybe_reweight_probas(self, probas: np.ndarray) -> np.ndarray:
-        """Reweights the probabilities if a target_metric is specified.
-
-        If a target metric is specified, the probabilities are reweighted based on
-        the true holdout sets labels and predicted logits. This is done to tune the
-        threshold for classification to the specified target metric.
-
-        Args:
-            probas: The predicted probabilities of the classes as a NumPy array.
-                Shape (n_samples, n_classes).
-
-        Returns:
-            The input probas if no tuning is done, otherwise the reweighted
-            probabilities.
-        """
-        if self.tuned_classification_thresholds_ is None:
-            return probas
-
-        probas = probas / np.maximum(self.tuned_classification_thresholds_, 1e-8)
-        return probas / probas.sum(axis=1, keepdims=True)
+        return output / output.sum(axis=1, keepdims=True)  # type: ignore
 
     def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
         """Scales logits by the softmax temperature."""
-        if self.softmax_temperature_ != 1.0:
-            return logits / self.softmax_temperature_
+        if self.softmax_temperature != 1.0:
+            return logits / self.softmax_temperature
         return logits
 
     def _average_across_estimators(self, tensors: torch.Tensor) -> torch.Tensor:
@@ -1180,89 +748,17 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
 
     def _apply_balancing(self, probas: torch.Tensor) -> torch.Tensor:
         """Applies class balancing to a probability tensor."""
-        return balance_probas_by_class_counts(probas, self.class_counts_)
-
-    def logits_to_probabilities(
-        self,
-        raw_logits: np.ndarray | torch.Tensor,
-        *,
-        softmax_temperature: float | None = None,
-        average_before_softmax: bool | None = None,
-        balance_probabilities: bool | None = None,
-    ) -> torch.Tensor:
-        """Convert logits to probabilities using the classifier's post-processing.
-
-        Args:
-            raw_logits: Logits with shape (n_estimators, n_samples, n_classes) or
-                (n_samples, n_classes). If the logits have three dimensions, they are
-                averaged across the estimator dimension (dim=0).
-            softmax_temperature: Optional override for temperature scaling.
-            average_before_softmax: Optional override for averaging order.
-            balance_probabilities: Optional override for probability balancing.
-
-        Returns:
-            Probabilities with shape (n_samples, n_classes).
-        """
-        raw_logits = (
-            raw_logits
-            if isinstance(raw_logits, torch.Tensor)
-            else torch.from_numpy(np.asarray(raw_logits))
-        )
-        used_temperature = (
-            softmax_temperature
-            if softmax_temperature is not None
-            else getattr(self, "softmax_temperature_", self.softmax_temperature)
-        )
-        use_average_before_softmax = (
-            self.average_before_softmax
-            if average_before_softmax is None
-            else average_before_softmax
-        )
-        use_balance = (
-            self.balance_probabilities
-            if balance_probabilities is None
-            else balance_probabilities
-        )
-
-        steps: list[Callable[[torch.Tensor], torch.Tensor]] = []
-
-        if used_temperature != 1.0:
-
-            def apply_temp(t: torch.Tensor) -> torch.Tensor:
-                return t / used_temperature
-
-            steps.append(apply_temp)
-
-        if raw_logits.ndim >= 3:
-            if use_average_before_softmax:
-                steps.append(self._average_across_estimators)
-                steps.append(self._apply_softmax)
-            else:
-                steps.append(self._apply_softmax)
-                steps.append(self._average_across_estimators)
-        elif raw_logits.ndim == 2:
-            steps.append(self._apply_softmax)
-        else:
-            raise ValueError(
-                f"Expected logits with 2 or more dims, got {raw_logits.ndim}"
-            )
-
-        if use_balance:
-            steps.append(self._apply_balancing)
-
-        output = raw_logits
-        for fn in steps:
-            output = fn(output)
-
-        return output
+        class_prob_in_train = self.class_counts_ / self.class_counts_.sum()
+        balanced_probas = probas / torch.Tensor(class_prob_in_train).to(self.device_)
+        return balanced_probas / balanced_probas.sum(dim=-1, keepdim=True)
 
+    # TODO: reduce complexity to remove noqa C901, PLR0912
     def forward(  # noqa: C901, PLR0912
         self,
         X: list[torch.Tensor] | torch.Tensor,
         *,
         use_inference_mode: bool = False,
         return_logits: bool = False,
-        return_raw_logits: bool = False,
     ) -> torch.Tensor:
         """Forward pass returning predicted probabilities or logits
         for TabPFNClassifier Inference Engine. Used in
@@ -1277,23 +773,14 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             use_inference_mode: Flag for inference mode., default at False since
             it is called within predict. During FineTuning forward() is called
             directly by user, so default should be False here.
-            return_logits: If True, returns logits averaged across estimators.
-                Otherwise, probabilities are returned.
-            return_raw_logits: If True, returns the raw logits, without
-                averaging estimators or temperature scaling.
+            return_logits: If True, returns raw logits. Otherwise, probabilities.
 
         Returns:
             The predicted probabilities or logits of the classes as a torch.Tensor.
             - If `use_inference_mode` is True: Shape (N_samples, N_classes)
             - If `use_inference_mode` is False (e.g., for training/fine-tuning):
               Shape (Batch_size, N_classes, N_samples), suitable for NLLLoss.
-            - If `return_raw_logits` is True: Shape (n_estimators, n_samples, n_classes)
         """
-        if return_logits and return_raw_logits:
-            raise ValueError(
-                "Cannot return both logits and raw logits. Please specify only one."
-            )
-
         # Scenario 1: Standard inference path
         is_standard_inference = use_inference_mode and not isinstance(
             self.executor_, InferenceEngineBatchedNoPreprocessing
@@ -1330,7 +817,7 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
         outputs = []
         for output, config in self.executor_.iter_outputs(
             X,
-            devices=self.devices_,
+            device=self.device_,
             autocast=self.use_autocast_,
         ):
             original_ndim = output.ndim
@@ -1372,20 +859,38 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
 
             outputs.append(torch.stack(output_batch, dim=1))
 
-        # --- Post-processing ---
+        # --- Post-processing Pipeline ---
+        # 'outputs' contains the raw, unscaled logits from each estimator.
         stacked_outputs = torch.stack(outputs)
 
+        # --- Build the processing pipeline by composing the steps in order ---
+        # The first step is always to apply the temperature scaling.
+        pipeline = [self._apply_temperature]
+
         if return_logits:
-            temp_scaled = self._apply_temperature(stacked_outputs)
-            output = self._average_across_estimators(temp_scaled)
-        elif return_raw_logits:
-            output = stacked_outputs
+            # For logits, we just average the temperature-scaled logits.
+            pipeline.append(self._average_across_estimators)
         else:
-            output = self.logits_to_probabilities(stacked_outputs)
+            # For probabilities, the order of averaging and softmax is crucial.
+            if self.average_before_softmax:
+                pipeline.extend([self._average_across_estimators, self._apply_softmax])
+            else:  # Average after softmax
+                pipeline.extend([self._apply_softmax, self._average_across_estimators])
+
+            # Balancing is the final optional step for probabilities.
+            if self.balance_probabilities:
+                pipeline.append(self._apply_balancing)
+
+        # --- Execute the pipeline ---
+        # Start with the initial raw logits
+        output = stacked_outputs
+        # Sequentially apply each function in the pipeline
+        for step_function in pipeline:
+            output = step_function(output)
 
         # --- Final output shaping ---
         if output.ndim > 2 and use_inference_mode:
-            output = output.squeeze(1) if not return_raw_logits else output.squeeze(2)
+            output = output.squeeze(1)
 
         if not use_inference_mode:
             # This case is primarily for fine-tuning where NLLLoss expects [B, C, N]
@@ -1415,7 +920,7 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
             np.ndarray
                 The computed embeddings for each fitted estimator.
         """
-        return get_embeddings(self, X, data_source)
+        return _get_embeddings(self, X, data_source)
 
     def save_fit_state(self, path: Path | str) -> None:
         """Save a fitted classifier, light wrapper around save_fitted_tabpfn_model."""
@@ -1432,19 +937,3 @@ class TabPFNClassifier(ClassifierMixin, BaseEstimator):
                 f"Attempting to load a '{est.__class__.__name__}' as '{cls.__name__}'"
             )
         return est
-
-
-def _validate_eval_metric(
-    eval_metric: str | ClassifierEvalMetrics | None,
-) -> ClassifierEvalMetrics:
-    if eval_metric is None:
-        return DEFAULT_CLASSIFICATION_EVAL_METRIC
-    if isinstance(eval_metric, ClassifierEvalMetrics):
-        return eval_metric
-    try:
-        return ClassifierEvalMetrics(eval_metric)  # Convert string to Enum
-    except ValueError as err:
-        valid_values = [e.value for e in ClassifierEvalMetrics]
-        raise ValueError(
-            f"Invalid eval_metric: `{eval_metric}`. Must be one of {valid_values}"
-        ) from err
diff --git a/src/tabpfn/inference_config.py b/src/tabpfn/config.py
similarity index 53%
rename from src/tabpfn/inference_config.py
rename to src/tabpfn/config.py
index 01b98e5..1849967 100644
--- a/src/tabpfn/inference_config.py
+++ b/src/tabpfn/config.py
@@ -1,63 +1,27 @@
-"""Additional configuration options for inference."""
+"""Configuration for the model interfaces."""
 
 #  Copyright (c) Prior Labs GmbH 2025.
 
 from __future__ import annotations
 
-import dataclasses
 from copy import deepcopy
+from dataclasses import dataclass
 from typing import Literal
 
-import pydantic
+from tabpfn.preprocessing import PreprocessorConfig
 
-from tabpfn.constants import ModelVersion, TaskType
-from tabpfn.preprocessing import (
-    PreprocessorConfig,
-    default_classifier_preprocessor_configs,
-    default_regressor_preprocessor_configs,
-    v2_5_classifier_preprocessor_configs,
-    v2_5_regressor_preprocessor_configs,
-    v2_classifier_preprocessor_configs,
-    v2_regressor_preprocessor_configs,
-)
 
+@dataclass
+class ModelInterfaceConfig:
+    """Constants used as default HPs in the model interfaces.
 
-# By default Pydantic dataclasses will ignore unrecognised config items, extra="forbid"
-# will raise an exception instead.
-@pydantic.dataclasses.dataclass(config=pydantic.ConfigDict(extra="forbid"))
-class InferenceConfig:
-    """Additional configuration options for inference.
-
-    Several configuration options for inference are exposed in the `TabPFNClassifier`
-    and `TabPFNRegressor` interfaces. The options in this class are more advanced and
-    not expected to be changed by the (standard) user.
+    These constants are not exposed to the models' init on purpose
+    to reduce the complexity for users. Furthermore, most of these
+    should not be optimized over by the (standard) user.
 
     Several of the preprocessing options are supported by our code for efficiency
     reasons (to avoid loading TabPFN multiple times). However, these can also be
     applied outside of the model interface.
-
-    This class must be serializable as it is peristed in the model checkpoints.
-
-    Do not edit the default values in this class, as this can affect the backwards
-    compatibility of the model checkpoints. Instead, edit `get_default()`.
-    """
-
-    PREPROCESS_TRANSFORMS: list[PreprocessorConfig]
-    """The preprocessing applied to the data before passing it to TabPFN. See
-    `PreprocessorConfig` for options and more details. If multiple `PreprocessorConfig`
-    are provided, they are (repeatedly) applied across different estimators.
-
-    By default, for classification, two preprocessors are applied:
-        1. Uses the original input data, all features transformed with a quantile
-            scaler, and the first n-many components of SVD transformer (whereby
-            n is a fract of on the number of features or samples). Categorical features
-            are ordinal encoded but all categories with less than 10 features are
-            ignored.
-        2. Uses the original input data, with categorical features as ordinal encoded.
-
-    By default, for regression, two preprocessor are applied:
-        1. The same as for classification, with a minimal different quantile scaler.
-        2. The original input data power transformed and categories onehot encoded.
     """
 
     MAX_UNIQUE_FOR_CATEGORICAL_FEATURES: int = 30
@@ -127,7 +91,27 @@ class InferenceConfig:
         - If a float, the percentage of samples to subsample.
     """
 
-    REGRESSION_Y_PREPROCESS_TRANSFORMS: tuple[str | None, ...] = (None, "safepower")
+    PREPROCESS_TRANSFORMS: list[PreprocessorConfig | dict] | None = None
+    """The preprocessing applied to the data before passing it to TabPFN. See
+    `PreprocessorConfig` for options and more details. If a list of `PreprocessorConfig`
+    is provided, the preprocessors are (repeatedly) applied across different estimators.
+
+    By default, for classification, two preprocessors are applied:
+        1. Uses the original input data, all features transformed with a quantile
+            scaler, and the first n-many components of SVD transformer (whereby
+            n is a fract of on the number of features or samples). Categorical features
+            are ordinal encoded but all categories with less than 10 features are
+            ignored.
+        2. Uses the original input data, with categorical features as ordinal encoded.
+
+    By default, for regression, two preprocessor are applied:
+        1. The same as for classification, with a minimal different quantile scaler.
+        2. The original input data power transformed and categories onehot encoded.
+    """
+    REGRESSION_Y_PREPROCESS_TRANSFORMS: tuple[
+        Literal["safepower", "power", "quantile_norm", None],
+        ...,
+    ] = (None, "safepower")
     """The preprocessing applied to the target variable before passing it to TabPFN for
     regression. This can be understood as scaling the target variable to better predict
     it. The preprocessors should be passed as a tuple/list and are then (repeatedly)
@@ -137,9 +121,11 @@ class InferenceConfig:
     more than one estimator).
 
     The options are:
-        - None: no preprocessing is done.
-        - One of the options from
-          `tabpfn.preprocessors.get_all_reshape_feature_distribution_preprocessors()`
+        - If None, no preprocessing is done.
+        - If "power", a power transformation is applied.
+        - If "safepower", a power transformation is applied with a safety factor to
+            avoid numerical issues.
+        - If "quantile_norm", a quantile normalization is applied.
     """
 
     USE_SKLEARN_16_DECIMAL_PRECISION: bool = False
@@ -150,6 +136,7 @@ class InferenceConfig:
      To improve reproducibility,set `._sklearn_16_decimal_precision = True` before
      calling `.predict()` or `.predict_proba()`."""
 
+    # TODO: move this somewhere else to support that this might change.
     MAX_NUMBER_OF_CLASSES: int = 10
     """The number of classes seen during pretraining for classification. If the
     number of classes is larger than this number, TabPFN requires an additional step
@@ -173,114 +160,40 @@ class InferenceConfig:
     _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD: None = None
     _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD: float = 12.0
 
-    def override_with_user_input(
-        self, user_config: dict | InferenceConfig | None
-    ) -> InferenceConfig:
-        """Return a new config with fields specified in `user_config` overwritten.
-
-        Args:
-            user_config: Config provided by the user at inference time.
-                If a dictionary, then the keys must match attributes of
-                    `InferenceConfig` and will be used to override these attributes.
-                If an `InferenceConfig` object, then the whole config is overridden with
-                    the values from the user config.
-                If None, then a copy of this config is returned with no fields changed.
-        """
-        if user_config is None:
-            return deepcopy(self)
-        if isinstance(user_config, InferenceConfig):
-            return deepcopy(user_config)
-        if isinstance(user_config, dict):
-            return dataclasses.replace(self, **user_config)
-        raise ValueError(
-            f"{user_config=}\nUnknown user config provided, see config above."
-        )
-
-    @classmethod
-    def get_default(
-        cls, task_type: TaskType, model_version: ModelVersion | Literal["latest"]
-    ) -> InferenceConfig:
-        """Return the default config for the given model version and task type.
-
-        For model versions after v2.5, the inference config is generated by calling this
-        function with `model_version=latest` and stored in the checkpoint. This stored
-        config is then loaded and used for inference.
-
-        For v2 and v2.5, the config is not stored in the checkpoint. Thus, for backwards
-        compatiblity, we define the v2 and v2.5 configs here and use those during
-        inference.
+    @staticmethod
+    def from_user_input(
+        *,
+        inference_config: dict | ModelInterfaceConfig | None,
+    ) -> ModelInterfaceConfig:
+        """Converts the user input to a `ModelInterfaceConfig` object.
+
+        The input inference_config can be a dictionary, a `ModelInterfaceConfig` object,
+        or None. If a dictionary is passed, the keys must match the attributes of
+        `ModelInterfaceConfig`. If a `ModelInterfaceConfig` object is passed, it is
+        returned as is. If None is passed, a new `ModelInterfaceConfig` object is
+        created with default values.
         """
-        if model_version == ModelVersion.V2:
-            if task_type == "multiclass":
-                return _get_v2_config(v2_classifier_preprocessor_configs())
-            if task_type == "regression":
-                return _get_v2_config(v2_regressor_preprocessor_configs())
-        if model_version == ModelVersion.V2_5:
-            if task_type == "multiclass":
-                return _get_v2_5_config(v2_5_classifier_preprocessor_configs())
-            if task_type == "regression":
-                return _get_v2_5_config(v2_5_regressor_preprocessor_configs())
-
-        if task_type == "multiclass":
-            return InferenceConfig(
-                PREPROCESS_TRANSFORMS=default_classifier_preprocessor_configs(),
-                MAX_NUMBER_OF_FEATURES=2000,
-                MAX_NUMBER_OF_SAMPLES=50_000,
-            )
-        if task_type == "regression":
-            return InferenceConfig(
-                PREPROCESS_TRANSFORMS=default_regressor_preprocessor_configs(),
-                MAX_NUMBER_OF_FEATURES=2000,
-                MAX_NUMBER_OF_SAMPLES=50_000,
-            )
-        raise ValueError(f"Unknown {task_type=} {model_version=}")
-
-
-def _get_v2_config(
-    preprocessor_configs: list[PreprocessorConfig],
-) -> InferenceConfig:
-    return InferenceConfig(
-        MAX_UNIQUE_FOR_CATEGORICAL_FEATURES=30,
-        MIN_UNIQUE_FOR_NUMERICAL_FEATURES=4,
-        MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE=100,
-        OUTLIER_REMOVAL_STD="auto",
-        FEATURE_SHIFT_METHOD="shuffle",
-        CLASS_SHIFT_METHOD="shuffle",
-        FINGERPRINT_FEATURE=True,
-        POLYNOMIAL_FEATURES="no",
-        SUBSAMPLE_SAMPLES=None,
-        PREPROCESS_TRANSFORMS=preprocessor_configs,
-        REGRESSION_Y_PREPROCESS_TRANSFORMS=(None, "safepower"),
-        USE_SKLEARN_16_DECIMAL_PRECISION=False,
-        MAX_NUMBER_OF_CLASSES=10,
-        MAX_NUMBER_OF_FEATURES=500,
-        MAX_NUMBER_OF_SAMPLES=10_000,
-        FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM=True,
-        _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD=None,
-        _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD=12.0,
-    )
-
-
-def _get_v2_5_config(
-    preprocessor_configs: list[PreprocessorConfig],
-) -> InferenceConfig:
-    return InferenceConfig(
-        MAX_UNIQUE_FOR_CATEGORICAL_FEATURES=30,
-        MIN_UNIQUE_FOR_NUMERICAL_FEATURES=4,
-        MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE=100,
-        OUTLIER_REMOVAL_STD="auto",
-        FEATURE_SHIFT_METHOD="shuffle",
-        CLASS_SHIFT_METHOD="shuffle",
-        FINGERPRINT_FEATURE=True,
-        POLYNOMIAL_FEATURES="no",
-        SUBSAMPLE_SAMPLES=None,
-        PREPROCESS_TRANSFORMS=preprocessor_configs,
-        REGRESSION_Y_PREPROCESS_TRANSFORMS=(None, "safepower"),
-        USE_SKLEARN_16_DECIMAL_PRECISION=False,
-        MAX_NUMBER_OF_CLASSES=10,
-        MAX_NUMBER_OF_FEATURES=2000,
-        MAX_NUMBER_OF_SAMPLES=50_000,
-        FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM=True,
-        _REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD=None,
-        _CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD=12.0,
-    )
+        if inference_config is None:
+            interface_config_ = ModelInterfaceConfig()
+        elif isinstance(inference_config, ModelInterfaceConfig):
+            interface_config_ = deepcopy(inference_config)
+        elif isinstance(inference_config, dict):
+            interface_config_ = ModelInterfaceConfig()
+            for key, value in inference_config.items():
+                if not hasattr(interface_config_, key):
+                    raise ValueError(
+                        f"Unknown kwarg passed to model construction: {key}",
+                    )
+                setattr(interface_config_, key, value)
+        else:
+            raise ValueError(f"Unknown {inference_config=} passed to model.")
+
+        if interface_config_.PREPROCESS_TRANSFORMS is not None:
+            interface_config_.PREPROCESS_TRANSFORMS = [
+                PreprocessorConfig.from_dict(config)
+                if isinstance(config, dict)
+                else config
+                for config in interface_config_.PREPROCESS_TRANSFORMS
+            ]
+
+        return interface_config_
diff --git a/src/tabpfn/constants.py b/src/tabpfn/constants.py
index b77549e..6d59fc7 100644
--- a/src/tabpfn/constants.py
+++ b/src/tabpfn/constants.py
@@ -7,9 +7,7 @@
 # enumeration of things
 from __future__ import annotations
 
-import pathlib
-from enum import Enum
-from typing import Any, Literal, Union
+from typing import Any, Literal
 from typing_extensions import TypeAlias
 
 import joblib
@@ -25,23 +23,12 @@ SampleWeightType: TypeAlias = Any
 YType: TypeAlias = Any
 TODO_TYPE1: TypeAlias = str
 
-ModelPath: TypeAlias = Union[str, pathlib.Path]
-
-
-class ModelVersion(str, Enum):
-    """Version of the model."""
-
-    V2 = "v2"
-    V2_5 = "v2.5"
-
-
 NA_PLACEHOLDER = "__MISSING__"
 
 SKLEARN_16_DECIMAL_PRECISION = 16
 PROBABILITY_EPSILON_ROUND_ZERO = 1e-3
 REGRESSION_NAN_BORDER_LIMIT_UPPER = 1e3
 REGRESSION_NAN_BORDER_LIMIT_LOWER = -1e3
-REGRESSION_CONSTANT_TARGET_BORDER_EPSILON = 1e-5
 AUTOCAST_DTYPE_BYTE_SIZE = 2  # bfloat16
 DEFAULT_DTYPE_BYTE_SIZE = 4  # float32
 
diff --git a/src/tabpfn/finetune_utils.py b/src/tabpfn/finetune_utils.py
index 42b878b..264693e 100644
--- a/src/tabpfn/finetune_utils.py
+++ b/src/tabpfn/finetune_utils.py
@@ -39,28 +39,23 @@ def clone_model_for_evaluation(
     Returns:
         A new instance of the model class, ready for evaluation.
     """
-    if hasattr(original_model, "models_") and original_model.models_ is not None:
+    if hasattr(original_model, "model_") and original_model.model_ is not None:
         # Deep copy necessary components to avoid modifying the original trained model
-        # Since this is for the purpose of fine tuning, at the moment,
-        # we only ever copy the first model and config.
-        new_model_state = copy.deepcopy(original_model.models_[0])
-        new_architecture_config = copy.deepcopy(original_model.configs_[0])
-        new_inference_config = copy.deepcopy(original_model.inference_config_)
+        new_model_state = copy.deepcopy(original_model.model_)
+        new_config = copy.deepcopy(original_model.config_)
 
         model_spec_obj = None
         if isinstance(original_model, TabPFNClassifier):
             model_spec_obj = ClassifierModelSpecs(
                 model=new_model_state,
-                architecture_config=new_architecture_config,
-                inference_config=new_inference_config,
+                config=new_config,
             )
         elif isinstance(original_model, TabPFNRegressor):
             # Regressor also needs the distribution criterion copied
-            new_bar_dist = copy.deepcopy(original_model.znorm_space_bardist_)
+            new_bar_dist = copy.deepcopy(original_model.bardist_)
             model_spec_obj = RegressorModelSpecs(
                 model=new_model_state,
-                architecture_config=new_architecture_config,
-                inference_config=new_inference_config,
+                config=new_config,
                 norm_criterion=new_bar_dist,
             )
         else:
diff --git a/src/tabpfn/inference.py b/src/tabpfn/inference.py
index 6e9c811..c16e9cb 100644
--- a/src/tabpfn/inference.py
+++ b/src/tabpfn/inference.py
@@ -4,12 +4,11 @@
 
 from __future__ import annotations
 
-import itertools
+import contextlib
 from abc import ABC, abstractmethod
 from collections.abc import Iterator, Sequence
 from copy import deepcopy
 from dataclasses import dataclass
-from functools import partial
 from pathlib import Path
 from typing import TYPE_CHECKING, Literal
 from typing_extensions import override
@@ -18,18 +17,14 @@ import joblib
 import numpy as np
 import torch
 
-from tabpfn.architectures.base.memory import (
-    set_save_peak_memory,
-    should_save_peak_mem,
-)
-from tabpfn.parallel_execute import parallel_execute
+from tabpfn.architectures.base.memory import MemoryUsageEstimator
 from tabpfn.preprocessing import fit_preprocessing
 from tabpfn.utils import get_autocast_context
 
 if TYPE_CHECKING:
+    from tabpfn.architectures.base.preprocessing import SequentialFeatureTransformer
     from tabpfn.architectures.interface import Architecture
     from tabpfn.preprocessing import EnsembleConfig
-    from tabpfn.preprocessors import SequentialFeatureTransformer
 
 
 @dataclass
@@ -63,38 +58,32 @@ class InferenceEngine(ABC):
     InferenceEngineCachePreprocessing engines also support toggling
     `torch.use_torch_inference_mode` via `use_torch_inference_mode`
     to enable/disable gradient tracking during prediction.
-
-    Attributes:
-        save_peak_mem: Whether to save peak memory usage.
-        dtype_byte_size: The byte size of the dtype.
-        models: The models to use for inference.
     """
 
     save_peak_mem: bool | Literal["auto"] | float | int
     dtype_byte_size: int
-    models: list[Architecture]
+    ensemble_configs: Sequence[EnsembleConfig]
 
     @abstractmethod
     def iter_outputs(
         self,
         X: np.ndarray,
         *,
-        devices: Sequence[torch.device],
+        device: torch.device,
         autocast: bool,
     ) -> Iterator[tuple[torch.Tensor, EnsembleConfig]]:
-        """Iterate over the outputs of the model for each ensemble configuration.
+        """Iterate over the outputs of the model.
 
-        Depending on the InferenceEngine used, this will run the forward pass of the
-        model for each estimator.
+        One for each ensemble configuration that was used to initialize the executor.
 
         Args:
             X: The input data to make predictions on.
-            devices: The devices to run the model on.
+            device: The device to run the model on.
             autocast: Whether to use torch.autocast during inference.
         """
         ...
 
-    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
+    def use_torch_inference_mode(self, *, use_inference: bool):
         """Enable/Disable `torch.inference_mode`.
 
         Disabling allows backpropagation (gradients) but is slower and uses more
@@ -115,7 +104,7 @@ class InferenceEngine(ABC):
             "This inference engine does not support torch.inference_mode changes."
         )
 
-    def save_state_except_model_weights(self, path: str | Path) -> None:
+    def save_state_expect_model_weights(self, path: str | Path) -> None:
         """Persist the executor state to ``path`` without the model weights.
 
         The state is first moved to CPU so the resulting file can be loaded
@@ -123,7 +112,12 @@ class InferenceEngine(ABC):
         excluded to keep the file small and efficient.
         """
         state_copy = deepcopy(self)
-        state_copy.models = None  # type: ignore
+
+        # Decouple the large model weights before serialization
+        if hasattr(state_copy, "model"):
+            state_copy.model = None
+        if hasattr(state_copy, "models"):
+            state_copy.models = None  # For KV cache engine
 
         joblib.dump(state_copy, path)
 
@@ -144,11 +138,12 @@ class InferenceEngineOnDemand(InferenceEngine):
 
     X_train: np.ndarray
     y_train: np.ndarray
+    ensemble_configs: Sequence[EnsembleConfig]
     cat_ix: list[int]
     static_seed: int
-    n_preprocessing_jobs: int
+    n_workers: int
+    model: Architecture
     force_inference_dtype: torch.dtype | None
-    ensemble_configs: list[EnsembleConfig]
 
     @classmethod
     def prepare(
@@ -157,10 +152,10 @@ class InferenceEngineOnDemand(InferenceEngine):
         y_train: np.ndarray,
         *,
         cat_ix: list[int],
-        models: list[Architecture],
+        model: Architecture,
         ensemble_configs: Sequence[EnsembleConfig],
         rng: np.random.Generator,
-        n_preprocessing_jobs: int,
+        n_workers: int,
         dtype_byte_size: int,
         force_inference_dtype: torch.dtype | None,
         save_peak_mem: bool | Literal["auto"] | float | int,
@@ -171,10 +166,10 @@ class InferenceEngineOnDemand(InferenceEngine):
             X_train: The training data.
             y_train: The training target.
             cat_ix: The categorical indices.
-            models: The models to use.
+            model: The model to use.
             ensemble_configs: The ensemble configurations to use.
             rng: The random number generator.
-            n_preprocessing_jobs: The number of workers to use.
+            n_workers: The number of workers to use.
             dtype_byte_size: The byte size of the dtype.
             force_inference_dtype: The dtype to force inference to.
             save_peak_mem: Whether to save peak memory usage.
@@ -184,11 +179,11 @@ class InferenceEngineOnDemand(InferenceEngine):
         return cls(
             X_train=X_train,
             y_train=y_train,
-            ensemble_configs=list(ensemble_configs),
+            ensemble_configs=ensemble_configs,
             cat_ix=cat_ix,
-            models=models,
+            model=model,
             static_seed=static_seed,
-            n_preprocessing_jobs=n_preprocessing_jobs,
+            n_workers=n_workers,
             dtype_byte_size=dtype_byte_size,
             force_inference_dtype=force_inference_dtype,
             save_peak_mem=save_peak_mem,
@@ -199,103 +194,65 @@ class InferenceEngineOnDemand(InferenceEngine):
         self,
         X: np.ndarray,
         *,
-        devices: Sequence[torch.device],
+        device: torch.device,
         autocast: bool,
         only_return_standard_out: bool = True,
     ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
         rng = np.random.default_rng(self.static_seed)
-
-        sorted_ensemble_configs = sorted(
-            self.ensemble_configs,
-            key=lambda c: c._model_index,
-        )
-
-        preprocessed_data_iterator = fit_preprocessing(
-            configs=sorted_ensemble_configs,
+        itr = fit_preprocessing(
+            configs=self.ensemble_configs,
             X_train=self.X_train,
             y_train=self.y_train,
             random_state=rng,
             cat_ix=self.cat_ix,
-            n_preprocessing_jobs=self.n_preprocessing_jobs,
-            parallel_mode="in-order",
+            n_workers=self.n_workers,
+            parallel_mode="as-ready",
         )
 
-        save_peak_mem = should_save_peak_mem(
-            self.save_peak_mem,
-            X_train_shape=self.X_train.shape,
-            X_test_shape=X.shape,
-            devices=devices,
-            dtype_byte_size=self.dtype_byte_size,
-        )
+        self.model = self.model.to(device)
+        if self.force_inference_dtype is not None:
+            self.model = self.model.type(self.force_inference_dtype)
 
-        ensemble_configs, preprocessings = itertools.tee(preprocessed_data_iterator)
+        for config, preprocessor, X_train, y_train, cat_ix in itr:
+            X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)  # noqa: PLW2901
 
-        if self.force_inference_dtype is not None:
-            [model.type(self.force_inference_dtype) for model in self.models]
-
-        model_forward_functions = (
-            partial(
-                self._call_model,
-                X_train=X_train,
-                X_test=preprocessor.transform(X).X,
-                y_train=y_train,
-                cat_ix=cat_ix,
-                only_return_standard_out=only_return_standard_out,
-                autocast=autocast,
-                model_index=config._model_index,
-                save_peak_mem=save_peak_mem,
-            )
-            for config, preprocessor, X_train, y_train, cat_ix in preprocessings
-        )
-        outputs = parallel_execute(devices, model_forward_functions)
+            X_test = preprocessor.transform(X).X
+            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
 
-        for (config, _, _, _, _), output in zip(ensemble_configs, outputs):
-            yield _move_and_squeeze_output(output, devices[0]), config
+            X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
+            batched_cat_ix = [cat_ix]
+            y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)  # type: ignore  # noqa: PLW2901
 
-        [model.cpu() for model in self.models]
+            MemoryUsageEstimator.reset_peak_memory_if_required(
+                save_peak_mem=self.save_peak_mem,
+                model=self.model,
+                X=X_full,
+                cache_kv=False,
+                dtype_byte_size=self.dtype_byte_size,
+                device=device,
+                safety_factor=1.2,  # TODO(Arjun): make customizable
+            )
 
-    def _call_model(
-        self,
-        *,
-        device: torch.device,
-        is_parallel: bool,
-        X_train: torch.Tensor | np.ndarray,
-        X_test: torch.Tensor | np.ndarray,
-        y_train: torch.Tensor | np.ndarray,
-        cat_ix: list[int],
-        autocast: bool,
-        only_return_standard_out: bool,
-        model_index: int,
-        save_peak_mem: bool,
-    ) -> torch.Tensor | dict[str, torch.Tensor]:
-        """Execute a model forward pass on the provided device.
-
-        Note that several instances of this function may be executed in parallel in
-        different threads, one for each device in the system.
-        """
-        # If several estimators are being run in parallel, then each thread needs its
-        # own copy of the model so it can move it to its device.
-        model = (
-            deepcopy(self.models[model_index])
-            if is_parallel
-            else self.models[model_index]
-        )
-        model.to(device)
+            if self.force_inference_dtype is not None:
+                X_full = X_full.type(self.force_inference_dtype)
+                y_train = y_train.type(self.force_inference_dtype)  # type: ignore  # noqa: PLW2901
 
-        X_full, y_train = _prepare_model_inputs(
-            device, self.force_inference_dtype, X_train, X_test, y_train
-        )
-        batched_cat_ix = [cat_ix]
+            with (
+                get_autocast_context(device, enabled=autocast),
+                torch.inference_mode(),
+            ):
+                output = self.model(
+                    X_full,
+                    y_train,
+                    only_return_standard_out=only_return_standard_out,
+                    categorical_inds=batched_cat_ix,
+                )
 
-        set_save_peak_memory(model, enabled=save_peak_mem)
+            output = output if isinstance(output, dict) else output.squeeze(1)
 
-        with get_autocast_context(device, enabled=autocast), torch.inference_mode():
-            return model(
-                X_full,
-                y_train,
-                only_return_standard_out=only_return_standard_out,
-                categorical_inds=batched_cat_ix,
-            )
+            yield output, config
+
+        self.model = self.model.cpu()
 
 
 @dataclass
@@ -307,6 +264,7 @@ class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
             X_trains: The training data.
             y_trains    : The training target.
             cat_ix: The categorical indices.
+            model: The model to use.
             ensemble_configs: The ensemble configurations to use.
             force_inference_dtype: The dtype to force inference to.
             save_peak_mem: Whether to save peak memory usage.
@@ -316,7 +274,8 @@ class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
     X_trains: list[torch.Tensor]
     y_trains: list[torch.Tensor]
     cat_ix: list[list[list[int]]]
-    ensemble_configs: list[list[EnsembleConfig]]
+    model: Architecture
+    ensemble_configs: Sequence[EnsembleConfig]
     force_inference_dtype: torch.dtype | None
     inference_mode: bool
 
@@ -327,8 +286,8 @@ class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
         y_trains: list[torch.Tensor],
         *,
         cat_ix: list[list[list[int]]],
-        models: list[Architecture],
-        ensemble_configs: list[list[EnsembleConfig]],
+        model: Architecture,
+        ensemble_configs: Sequence[EnsembleConfig],
         force_inference_dtype: torch.dtype | None,
         inference_mode: bool,
         dtype_byte_size: int,
@@ -340,26 +299,19 @@ class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
             X_trains: The training data.
             y_trains: The training target.
             cat_ix: The categorical indices.
-            models: The models to use.
+            model: The model to use.
             ensemble_configs: The ensemble configurations to use.
             inference_mode: Whether to use torch inference mode.
             dtype_byte_size: The byte size of the dtype.
             force_inference_dtype: The dtype to force inference to.
             save_peak_mem: Whether to save peak memory usage.
         """
-        for ensemble_config in ensemble_configs:
-            if len(ensemble_config) > 1:
-                raise ValueError(
-                    "Batched inference does not support multiple ensemble"
-                    " configurations because no preprocessing is applied."
-                )
-
         # We save it as a static seed to be reproducible across predicts
         return cls(
             X_trains=X_trains,
             y_trains=y_trains,
             cat_ix=cat_ix,
-            models=models,
+            model=model,
             ensemble_configs=ensemble_configs,
             force_inference_dtype=force_inference_dtype,
             inference_mode=inference_mode,
@@ -372,15 +324,12 @@ class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
         self,
         X: list[torch.Tensor],
         *,
-        devices: Sequence[torch.device],
+        device: torch.device,
         autocast: bool,
-    ) -> Iterator[tuple[torch.Tensor | dict, list[EnsembleConfig]]]:
-        # This engine currently only supports one device, so just take the first.
-        device = devices[0]
-
-        self.models = [model.to(device) for model in self.models]
-        batch_size = len(self.X_trains)
-        for i in range(batch_size):
+    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
+        self.model = self.model.to(device)
+        ensemble_size = len(self.X_trains)
+        for i in range(ensemble_size):
             train_x_full = torch.cat([self.X_trains[i], X[i]], dim=-2)
             train_y_batch = self.y_trains[i]
             train_x_full = train_x_full.to(device)
@@ -390,10 +339,10 @@ class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
                 train_y_batch = train_y_batch.type(self.force_inference_dtype)  # type: ignore
 
             with (
-                get_autocast_context(device, enabled=autocast),
+                torch.autocast(device.type, enabled=autocast),
                 torch.inference_mode(self.inference_mode),
             ):
-                output = self.models[self.ensemble_configs[i][0]._model_index](
+                output = self.model(
                     train_x_full.transpose(0, 1),
                     train_y_batch.transpose(0, 1),
                     only_return_standard_out=True,
@@ -402,10 +351,10 @@ class InferenceEngineBatchedNoPreprocessing(InferenceEngine):
 
             yield output, self.ensemble_configs[i]
         if self.inference_mode:  ## if inference
-            [model.cpu() for model in self.models]
+            self.model = self.model.cpu()
 
     @override
-    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
+    def use_torch_inference_mode(self, use_inference: bool):
         self.inference_mode = use_inference
 
 
@@ -424,12 +373,12 @@ class InferenceEngineCachePreprocessing(InferenceEngine):
 
     X_trains: Sequence[np.ndarray | torch.Tensor]
     y_trains: Sequence[np.ndarray | torch.Tensor]
-    X_train_shape_before_preprocessing: tuple[int, int]
     cat_ixs: Sequence[list[int]]
+    ensemble_configs: Sequence[EnsembleConfig]
     preprocessors: Sequence[SequentialFeatureTransformer]
+    model: Architecture
     force_inference_dtype: torch.dtype | None
     inference_mode: bool
-    ensemble_configs: list[EnsembleConfig]
     no_preprocessing: bool = False
 
     @classmethod
@@ -439,9 +388,9 @@ class InferenceEngineCachePreprocessing(InferenceEngine):
         y_train: np.ndarray | torch.Tensor,
         *,
         cat_ix: list[int],
-        models: list[Architecture],
+        model: Architecture,
         ensemble_configs: Sequence[EnsembleConfig],
-        n_preprocessing_jobs: int,
+        n_workers: int,
         rng: np.random.Generator,
         dtype_byte_size: int,
         force_inference_dtype: torch.dtype | None,
@@ -455,9 +404,9 @@ class InferenceEngineCachePreprocessing(InferenceEngine):
             X_train: The training data.
             y_train: The training target.
             cat_ix: The categorical indices.
-            models: The models to use.
+            model: The model to use.
             ensemble_configs: The ensemble configurations to use.
-            n_preprocessing_jobs: The number of workers to use.
+            n_workers: The number of workers to use.
             rng: The random number generator.
             dtype_byte_size: The byte size of the dtype.
             force_inference_dtype: The dtype to force inference to.
@@ -476,15 +425,14 @@ class InferenceEngineCachePreprocessing(InferenceEngine):
             y_train=y_train,
             random_state=rng,
             cat_ix=cat_ix,
-            n_preprocessing_jobs=n_preprocessing_jobs,
+            n_workers=n_workers,
             parallel_mode="block",
         )
         configs, preprocessors, X_trains, y_trains, cat_ixs = list(zip(*itr))
         return InferenceEngineCachePreprocessing(
             X_trains=X_trains,
             y_trains=y_trains,
-            X_train_shape_before_preprocessing=tuple[int, int](X_train.shape),
-            models=models,
+            model=model,
             cat_ixs=cat_ixs,
             ensemble_configs=configs,
             preprocessors=preprocessors,
@@ -498,106 +446,75 @@ class InferenceEngineCachePreprocessing(InferenceEngine):
     @override
     def iter_outputs(
         self,
-        X: np.ndarray | torch.Tensor,
+        X: np.ndarray | torch.tensor,
         *,
-        devices: Sequence[torch.device],
+        device: torch.device,
         autocast: bool,
         only_return_standard_out: bool = True,
     ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
+        self.model = self.model.to(device)
         if self.force_inference_dtype is not None:
-            [model.type(self.force_inference_dtype) for model in self.models]
-
-        if self.inference_mode:
-            save_peak_mem = should_save_peak_mem(
-                memory_saving_mode=self.save_peak_mem,
-                X_train_shape=self.X_train_shape_before_preprocessing,
-                X_test_shape=tuple[int, int](X.shape),
-                devices=devices,
-                dtype_byte_size=self.dtype_byte_size,
-            )
-        else:
-            save_peak_mem = False
-
-        # Create a sorted index order by model_index so that calls with model index 0
-        # run first, then model index 1, etc.
-        sorted_indices = sorted(
-            range(len(self.ensemble_configs)),
-            key=lambda i: self.ensemble_configs[i]._model_index,
-        )
-
-        def _transform_X_test(i: int) -> np.ndarray | torch.Tensor:
-            return X if self.no_preprocessing else self.preprocessors[i].transform(X).X
-
-        model_forward_functions = (
-            partial(
-                self._call_model,
-                X_train=self.X_trains[i],
-                X_test=_transform_X_test(i),
-                y_train=self.y_trains[i],
-                cat_ix=self.cat_ixs[i],
-                autocast=autocast,
-                only_return_standard_out=only_return_standard_out,
-                model_index=self.ensemble_configs[i]._model_index,
-                save_peak_mem=save_peak_mem,
-            )
-            for i in sorted_indices
-        )
-        outputs = parallel_execute(devices, model_forward_functions)
-
-        for output, i in zip(outputs, sorted_indices):
-            yield _move_and_squeeze_output(output, devices[0]), self.ensemble_configs[i]
+            self.model = self.model.type(self.force_inference_dtype)
+        for preprocessor, X_train, y_train, config, cat_ix in zip(
+            self.preprocessors,
+            self.X_trains,
+            self.y_trains,
+            self.ensemble_configs,
+            self.cat_ixs,
+        ):
+            if not isinstance(X_train, torch.Tensor):
+                X_train = torch.as_tensor(X_train, dtype=torch.float32)  # noqa: PLW2901
+            X_train = X_train.to(device)  # noqa: PLW2901
+            X_test = preprocessor.transform(X).X if not self.no_preprocessing else X
+            if not isinstance(X_test, torch.Tensor):
+                X_test = torch.as_tensor(X_test, dtype=torch.float32)
+            X_test = X_test.to(device)
+            X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
+            if not isinstance(y_train, torch.Tensor):
+                y_train = torch.as_tensor(y_train, dtype=torch.float32)  # noqa: PLW2901
+            y_train = y_train.to(device)  # noqa: PLW2901
 
-        if self.inference_mode:
-            [model.cpu() for model in self.models]
+            batched_cat_ix = [cat_ix]
 
-    def _call_model(
-        self,
-        *,
-        device: torch.device,
-        is_parallel: bool,
-        X_train: torch.Tensor | np.ndarray,
-        X_test: torch.Tensor | np.ndarray,
-        y_train: torch.Tensor | np.ndarray,
-        cat_ix: list[int],
-        autocast: bool,
-        only_return_standard_out: bool,
-        model_index: int,
-        save_peak_mem: bool,
-    ) -> torch.Tensor | dict[str, torch.Tensor]:
-        """Execute a model forward pass on the provided device.
-
-        Note that several instances of this function may be executed in parallel in
-        different threads, one for each device in the system.
-        """
-        # If several estimators are being run in parallel, then each thread needs its
-        # own copy of the model so it can move it to its device.
-        model = (
-            deepcopy(self.models[model_index])
-            if is_parallel
-            else self.models[model_index]
-        )
-        model.to(device)
+            # Handle type casting
+            with contextlib.suppress(Exception):  # Avoid overflow error
+                X_full = X_full.float()
+            if self.force_inference_dtype is not None:
+                X_full = X_full.type(self.force_inference_dtype)
+                y_train = y_train.type(self.force_inference_dtype)  # type: ignore # noqa: PLW2901
+
+            if self.inference_mode:
+                MemoryUsageEstimator.reset_peak_memory_if_required(
+                    save_peak_mem=self.save_peak_mem,
+                    model=self.model,
+                    X=X_full,
+                    cache_kv=False,
+                    device=device,
+                    dtype_byte_size=self.dtype_byte_size,
+                    safety_factor=1.2,  # TODO(Arjun): make customizable
+                )
+            else:
+                pass
 
-        set_save_peak_memory(model, enabled=save_peak_mem)
+            with (
+                get_autocast_context(device, enabled=autocast),
+                torch.inference_mode(self.inference_mode),
+            ):
+                output = self.model(
+                    X_full,
+                    y_train,
+                    only_return_standard_out=only_return_standard_out,
+                    categorical_inds=batched_cat_ix,
+                )
 
-        X_full, y_train = _prepare_model_inputs(
-            device, self.force_inference_dtype, X_train, X_test, y_train
-        )
-        batched_cat_ix = [cat_ix]
+            output = output if isinstance(output, dict) else output.squeeze(1)
 
-        with (
-            get_autocast_context(device, enabled=autocast),
-            torch.inference_mode(self.inference_mode),
-        ):
-            return model(
-                X_full,
-                y_train,
-                only_return_standard_out=only_return_standard_out,
-                categorical_inds=batched_cat_ix,
-            )
+            yield output, config
+        if self.inference_mode:  ## if inference
+            self.model = self.model.cpu()
 
     @override
-    def use_torch_inference_mode(self, *, use_inference: bool) -> None:
+    def use_torch_inference_mode(self, use_inference: bool):
         self.inference_mode = use_inference
 
 
@@ -612,10 +529,11 @@ class InferenceEngineCacheKV(InferenceEngine):
     """
 
     preprocessors: list[SequentialFeatureTransformer]
+    ensemble_configs: list[EnsembleConfig]
     cat_ixs: Sequence[list[int]]
+    models: list[Architecture]
     n_train_samples: list[int]
     force_inference_dtype: torch.dtype | None
-    ensemble_configs: list[EnsembleConfig]
 
     @classmethod
     def prepare(  # noqa: PLR0913
@@ -625,9 +543,9 @@ class InferenceEngineCacheKV(InferenceEngine):
         *,
         cat_ix: list[int],
         ensemble_configs: Sequence[EnsembleConfig],
-        n_preprocessing_jobs: int,
-        models: list[Architecture],
-        devices: Sequence[torch.device],
+        n_workers: int,
+        model: Architecture,
+        device: torch.device,
         rng: np.random.Generator,
         dtype_byte_size: int,
         force_inference_dtype: torch.dtype | None,
@@ -642,9 +560,9 @@ class InferenceEngineCacheKV(InferenceEngine):
             y_train: The training target.
             cat_ix: The categorical indices.
             ensemble_configs: The ensemble configurations to use.
-            n_preprocessing_jobs: The number of workers to use.
-            models: The models to use.
-            devices: The devices to run the model on.
+            n_workers: The number of workers to use.
+            model: The model to use.
+            device: The device to run the model on.
             rng: The random number generator.
             dtype_byte_size: Size of the dtype in bytes.
             force_inference_dtype: The dtype to force inference to.
@@ -652,19 +570,16 @@ class InferenceEngineCacheKV(InferenceEngine):
             autocast: Whether to use torch.autocast during inference.
             only_return_standard_out: Whether to only return the standard output
         """
-        # This engine currently only supports one device, so just take the first.
-        device = devices[0]
-
         itr = fit_preprocessing(
             configs=ensemble_configs,
             X_train=X_train,
             y_train=y_train,
             random_state=rng,
             cat_ix=cat_ix,
-            n_preprocessing_jobs=n_preprocessing_jobs,
+            n_workers=n_workers,
             parallel_mode="as-ready",
         )
-        ens_models: list[Architecture] = []
+        models: list[Architecture] = []
         preprocessors: list[SequentialFeatureTransformer] = []
         correct_order_configs: list[EnsembleConfig] = []
         cat_ixs: Sequence[list[int]] = []
@@ -676,7 +591,7 @@ class InferenceEngineCacheKV(InferenceEngine):
             correct_order_configs.append(config)
             n_train_samples.append(len(y))
 
-            ens_model = deepcopy(models[config._model_index])
+            ens_model = deepcopy(model)
             ens_model = ens_model.to(device)
             if not isinstance(X, torch.Tensor):
                 X = torch.as_tensor(X, dtype=torch.float32, device=device)  # noqa: PLW2901
@@ -703,14 +618,14 @@ class InferenceEngineCacheKV(InferenceEngine):
             if device.type != "cpu":
                 ens_model = ens_model.cpu()
 
-            ens_models.append(ens_model)
+            models.append(ens_model)
 
         return InferenceEngineCacheKV(
             preprocessors=preprocessors,
             ensemble_configs=correct_order_configs,
             cat_ixs=cat_ixs,
             n_train_samples=n_train_samples,
-            models=ens_models,
+            models=models,
             dtype_byte_size=dtype_byte_size,
             force_inference_dtype=force_inference_dtype,
             save_peak_mem=save_peak_mem,
@@ -721,14 +636,11 @@ class InferenceEngineCacheKV(InferenceEngine):
         self,
         X: np.ndarray,
         *,
-        devices: Sequence[torch.device],
+        device: torch.device,
         autocast: bool,
         only_return_standard_out: bool = True,
     ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
-        # This engine currently only supports one device, so just take the first.
-        device = devices[0]
-
-        for preprocessor, model, config, cat_ix, _X_train_len in zip(
+        for preprocessor, model, config, cat_ix, X_train_len in zip(
             self.preprocessors,
             self.models,
             self.ensemble_configs,
@@ -740,10 +652,16 @@ class InferenceEngineCacheKV(InferenceEngine):
             X_test = X_test.unsqueeze(1)
             batched_cat_ix = [cat_ix]
 
-            # When the KV cache is enabled, we assume we are under memory pressure and
-            # enable the saving mode.
-            # TODO: Use the heuristic in this case also.
-            set_save_peak_memory(model, enabled=True)
+            MemoryUsageEstimator.reset_peak_memory_if_required(
+                save_peak_mem=self.save_peak_mem,
+                model=model,
+                X=X_test,
+                cache_kv=True,
+                device=device,
+                dtype_byte_size=self.dtype_byte_size,
+                safety_factor=1.2,  # TODO(Arjun): make customizable
+                n_train_samples=X_train_len,
+            )
 
             model = model.to(device)  # noqa: PLW2901
 
@@ -768,26 +686,3 @@ class InferenceEngineCacheKV(InferenceEngine):
             output = output if isinstance(output, dict) else output.squeeze(1)
 
             yield output, config
-
-
-def _prepare_model_inputs(
-    device: torch.device,
-    force_inference_dtype: torch.dtype | None,
-    X_train: torch.Tensor | np.ndarray,
-    X_test: torch.Tensor | np.ndarray,
-    y_train: torch.Tensor | np.ndarray,
-) -> tuple[torch.Tensor, torch.Tensor]:
-    dtype = force_inference_dtype if force_inference_dtype else torch.float32
-    X_train = torch.as_tensor(X_train, dtype=dtype, device=device)
-    X_test = torch.as_tensor(X_test, dtype=dtype, device=device)
-    X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
-    y_train = torch.as_tensor(y_train, dtype=dtype, device=device)
-    return X_full, y_train
-
-
-def _move_and_squeeze_output(
-    output: dict | torch.Tensor, device: torch.device
-) -> dict[str, torch.Tensor] | torch.Tensor:
-    if isinstance(output, dict):
-        return {k: v.to(device) for k, v in output.items()}
-    return output.squeeze(1).to(device)
diff --git a/src/tabpfn/inference_tuning.py b/src/tabpfn/inference_tuning.py
deleted file mode 100644
index fd3f4ea..0000000
--- a/src/tabpfn/inference_tuning.py
+++ /dev/null
@@ -1,417 +0,0 @@
-"""Inference tuning helpers for TabPFN fit/predict calls."""
-
-from __future__ import annotations
-
-import dataclasses
-import warnings
-from enum import Enum
-from typing import TYPE_CHECKING, Callable, Literal
-from typing_extensions import Self
-
-import numpy as np
-from sklearn.metrics import (
-    accuracy_score,
-    balanced_accuracy_score,
-    f1_score,
-    log_loss,
-    roc_auc_score,
-)
-from sklearn.model_selection import StratifiedKFold
-
-if TYPE_CHECKING:
-    import torch
-
-
-MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING = 500
-
-
-@dataclasses.dataclass
-class TuningConfig:
-    """Configuration for tuning the model during fit/predict calls."""
-
-    calibrate_temperature: bool = False
-    """Whether to calibrate the softmax temperature. Set to True to enable."""
-
-    tuning_holdout_frac: Literal["auto"] | float = "auto"
-    """The percentage of the data to hold out for tuning per split. If "auto", a value
-    is automatically chosen based on the dataset size, trading off between
-    computational cost and accuracy."""
-
-    tuning_n_folds: Literal["auto"] | int = "auto"
-    """The number of cross-validation folds to use for tuning. If "auto", a value
-    is automatically chosen based on the dataset size, trading off between
-    computational cost and accuracy."""
-
-    def resolve(self: Self, num_samples: int) -> Self:
-        """Resolves 'auto' values based on the number of samples.
-
-        Args:
-            num_samples: The number of samples in the training data.
-
-        Returns:
-            A new TuningConfig instance with resolved values.
-        """
-        tuning_holdout_frac = (
-            get_default_tuning_holdout_frac(n_samples=num_samples)
-            if self.tuning_holdout_frac == "auto"
-            else self.tuning_holdout_frac
-        )
-        tuning_n_folds = (
-            get_default_tuning_n_folds(n_samples=num_samples)
-            if self.tuning_n_folds == "auto"
-            else self.tuning_n_folds
-        )
-        return dataclasses.replace(
-            self,
-            tuning_holdout_frac=tuning_holdout_frac,
-            tuning_n_folds=tuning_n_folds,
-        )
-
-
-@dataclasses.dataclass
-class ClassifierTuningConfig(TuningConfig):
-    """Configuration for tuning the model during fit/predict calls
-    for classification tasks.
-    """
-
-    tune_decision_thresholds: bool = False
-    """Whether to tune decision thresholds for the specified `eval_metric`.
-    Set to True to enable."""
-
-
-class ClassifierEvalMetrics(str, Enum):
-    """Metric by which predictions will be ultimately evaluated on test data."""
-
-    F1 = "f1"
-    ACCURACY = "accuracy"
-    BALANCED_ACCURACY = "balanced_accuracy"
-    ROC_AUC = "roc_auc"
-    LOG_LOSS = "log_loss"
-
-
-METRIC_NAME_TO_OBJECTIVE = {
-    "f1": lambda y_true, y_pred: -f1_score(
-        y_true,
-        y_pred,
-        average="binary",
-        zero_division=0,
-    ),
-    "accuracy": lambda y_true, y_pred: -accuracy_score(
-        y_true,
-        y_pred,
-    ),
-    "balanced_accuracy": lambda y_true, y_pred: -balanced_accuracy_score(
-        y_true,
-        y_pred,
-    ),
-    "roc_auc": lambda y_true, y_pred: -roc_auc_score(
-        y_true,
-        y_pred,
-    ),
-    "log_loss": log_loss,
-}
-
-
-def compute_metric_to_minimize(
-    metric_name: ClassifierEvalMetrics,
-    y_true: np.ndarray,
-    y_pred: np.ndarray,
-) -> float:
-    """Computes the metric.
-
-    Adjusts the sign of the metric to frame the problem as a minimization
-    problem.
-    """
-    if metric_name not in METRIC_NAME_TO_OBJECTIVE:
-        raise ValueError(
-            f"Metric '{metric_name}' is not supported. "
-            f"Supported metrics are: {list(METRIC_NAME_TO_OBJECTIVE.keys())}"
-        )
-    return METRIC_NAME_TO_OBJECTIVE[metric_name](y_true, y_pred)
-
-
-def get_tuning_splits(
-    X: np.ndarray,
-    y: np.ndarray,
-    holdout_frac: float,
-    n_splits: int = 1,
-    random_state: int | np.random.RandomState | np.random.Generator | None = 0,
-) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
-    """Get stratified tuning split(s) for the given configuration.
-
-    Args:
-        X: The input data of shape [n_samples, n_features].
-        y: The target labels of shape [n_samples].
-        holdout_frac: The percentage of the data to hold out for tuning.
-        n_splits: Number of stratified random splits to generate.
-        random_state: The random state to use for the split(s).
-
-    Returns:
-        Returns a list of splits as tuples of
-        (X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh).
-        Shape suffixes: Nt=num train samples, F=num features, Nh=num holdout samples.
-    """
-    # We want to use StratifiedKFold to ensure that no train samples are used twice.
-    # Therefore, we have to invert the holdout_frac to get the number of folds to
-    # use for StratifiedKFold. Round holdout_frac to 2 digits to avoid needing
-    # more than 100 folds
-    rounded_holdout_frac = round(holdout_frac, 2)
-    n_folds = max(2, round(1 / rounded_holdout_frac))
-
-    splitter = StratifiedKFold(
-        n_splits=n_folds,
-        shuffle=True,
-        random_state=random_state,
-    )
-
-    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
-    for i, (train_indices, holdout_indices) in enumerate(splitter.split(X, y)):
-        if i >= n_splits:
-            break
-        X_train_NtF = X[train_indices]
-        X_holdout_NhF = X[holdout_indices]
-        y_train_Nt = y[train_indices]
-        y_holdout_Nh = y[holdout_indices]
-        splits.append((X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh))
-
-    return splits
-
-
-def find_optimal_classification_thresholds(
-    metric_name: ClassifierEvalMetrics,
-    y_true: np.ndarray,
-    y_pred_probas: np.ndarray,
-    n_classes: int,
-) -> np.ndarray:
-    """Finds the optimal thresholds for each class in a one-vs-rest (OvR) fashion.
-
-    Args:
-        metric_name: The name of the metric to optimize.
-        y_true: The true labels of shape [n_samples].
-        y_pred_probas: The predicted probabilities of shape [n_samples, n_classes].
-        n_classes: The number of classes.
-
-    Returns:
-        The optimal thresholds of shape [n_classes].
-    """
-    optimal_thresholds = []
-
-    # TODO: vectorize this loop loop and the one in
-    # find_optimal_classification_threshold_single_class.
-    for i in range(n_classes):
-        y_true_ovr = (y_true == i).astype(int)
-        y_pred_probas_ovr = y_pred_probas[:, i]
-        best_thresh = find_optimal_classification_threshold_single_class(
-            metric_name=metric_name,
-            y_true=y_true_ovr,
-            y_pred_probas=y_pred_probas_ovr,
-        )
-
-        optimal_thresholds.append(best_thresh)
-
-    return np.array(optimal_thresholds)
-
-
-def find_optimal_classification_threshold_single_class(
-    metric_name: ClassifierEvalMetrics,
-    y_true: np.ndarray,
-    y_pred_probas: np.ndarray,
-) -> float:
-    """Finds the optimal classification threshold to maximize the specified metric.
-
-    The true labels are binary, and the predicted probabilities are the probabilities of
-    the positive class.
-
-    Args:
-        metric_name: The name of the metric to optimize.
-        y_true: The true labels of shape [n_samples].
-        y_pred_probas: The predicted probabilities of shape [n_samples].
-
-    Returns:
-        The optimal threshold.
-    """
-    thresholds = np.linspace(0.01, 0.99, 198)
-    thresholds_and_losses: list[tuple[float, float]] = []  # (threshold, metric)
-
-    for threshold in thresholds:
-        y_pred_tuned = (y_pred_probas >= threshold).astype(int)
-        current_loss = compute_metric_to_minimize(
-            metric_name=metric_name,
-            y_true=y_true,
-            y_pred=y_pred_tuned,
-        )
-        thresholds_and_losses.append((float(threshold), current_loss))
-
-    return select_robust_optimal_threshold(thresholds_and_losses=thresholds_and_losses)
-
-
-def select_robust_optimal_threshold(
-    thresholds_and_losses: list[tuple[float, float]],
-    plateau_delta: float = 0.002,
-) -> float:
-    """Selects the robust optimal threshold for the given metric.
-
-    This method avoids selecting a threshold that is at the edge
-    of a plateau which may not generalize well.
-
-    Args:
-        thresholds_and_losses: The thresholds and losses as a list of tuples.
-            The first element of the tuple is the threshold and the second element
-            is the loss.
-        plateau_delta: The delta to define a plateau around the best loss.
-
-    Returns:
-        The robust optimal threshold.
-    """
-    thresholds = np.array([t for t, _ in thresholds_and_losses], dtype=float)
-    losses = np.array([f for _, f in thresholds_and_losses], dtype=float)
-    best_loss = float(np.min(losses))
-    close_mask = losses <= (best_loss + plateau_delta)
-
-    # Find the contiguous region around the global minimum index
-    min_loss_index = int(np.argmin(losses))
-    start = min_loss_index
-    while start - 1 >= 0 and close_mask[start - 1]:
-        start -= 1
-    end = min_loss_index
-    num_points = len(losses)
-    while end + 1 < num_points and close_mask[end + 1]:
-        end += 1
-    mid_index = (start + end) // 2
-    robust_threshold = float(thresholds[mid_index])
-
-    # Edge guard: if chosen threshold is at exact endpoints and region has width,
-    # pick the second point from edge
-    if mid_index == 0 and end > 0:
-        robust_threshold = float(thresholds[1])
-    elif mid_index == num_points - 1 and start < num_points - 1:
-        robust_threshold = float(thresholds[num_points - 2])
-
-    return robust_threshold
-
-
-def find_optimal_temperature(
-    raw_logits: np.ndarray,
-    y_true: np.ndarray,
-    logits_to_probabilities_fn: Callable[
-        [np.ndarray | torch.Tensor, float], np.ndarray
-    ],
-    current_default_temperature: float,
-) -> float:
-    """Finds the optimal temperature to maximize the specified metric.
-
-    Args:
-        raw_logits: The raw logits of shape [n_estimators, n_samples, n_classes].
-        y_true: The true labels of shape [n_samples].
-        logits_to_probabilities_fn: The function to convert logits to probabilities.
-            The function should take an array of the shape of raw_logits and
-            a softmax temperature as argument.
-        current_default_temperature: The current default temperature.
-
-    Returns:
-        The temperature that minimizes the log loss.
-    """
-    temperatures = np.linspace(0.6, 1.4, 82)
-    best_log_loss = float("inf")
-    best_temperature = current_default_temperature
-
-    # TODO: think about vectorizing this loop.
-    for temperature in temperatures:
-        probas = logits_to_probabilities_fn(raw_logits, temperature)
-        current_log_loss = log_loss(y_true=y_true, y_pred=probas)
-
-        if current_log_loss < best_log_loss:
-            best_log_loss = current_log_loss
-            best_temperature = temperature
-
-    return best_temperature
-
-
-def get_default_tuning_holdout_frac(n_samples: int) -> float:
-    """Gets the default tuning holdout percentage based on a heuristic.
-
-    We aim to tradeoff between computational cost and accuracy.
-    """
-    n_samples_to_holdout_frac = {
-        2_000: 0.1,
-        5_000: 0.2,
-        10_000: 0.2,
-        20_000: 0.2,
-        50_000: 0.3,
-    }
-    for n_samples_threshold, frac in n_samples_to_holdout_frac.items():
-        if n_samples <= n_samples_threshold:
-            return frac
-    return 0.2
-
-
-def get_default_tuning_n_folds(n_samples: int) -> int:
-    """Gets the default tuning holdout number of splits based on a heuristic.
-
-    We aim to tradeoff between computational cost and accuracy.
-    """
-    n_samples_to_n_folds = {
-        2_000: 10,
-        5_000: 5,
-        10_000: 3,
-        20_000: 2,
-        50_000: 1,
-    }
-    for n_samples_threshold, n_folds in n_samples_to_n_folds.items():
-        if n_samples <= n_samples_threshold:
-            return n_folds
-    return 1
-
-
-def resolve_tuning_config(
-    tuning_config: dict | ClassifierTuningConfig | None,
-    num_samples: int,
-) -> ClassifierTuningConfig | None:
-    """Resolves the tuning configuration by checking if tuning is needed,
-    resolving 'auto' values for holdout parameters, and returning the appropriate
-    type of tuning configuration if the input is a dict.
-
-    Args:
-        tuning_config: The tuning configuration to use. If a dict is provided,
-            the function will infer the appropriate config type based on the keys
-            present (e.g., 'tune_decision_thresholds' indicates
-            ClassificationTuningConfig).
-        num_samples: The number of samples in the training data.
-
-    Returns:
-        The resolved tuning configuration or None if no tuning is needed.
-        The returned type will be the same as the input type (or inferred from dict).
-    """
-    if tuning_config is None:
-        return None
-
-    tuning_config = (
-        ClassifierTuningConfig(**tuning_config)
-        if isinstance(tuning_config, dict)
-        else tuning_config
-    )
-
-    compute_holdout_logits = bool(
-        tuning_config.calibrate_temperature or tuning_config.tune_decision_thresholds
-    )
-    if not compute_holdout_logits:
-        warnings.warn(
-            "You specified a tuning configuration but no tuning features were enabled. "
-            "Set `calibrate_temperature=True` or `tune_decision_thresholds=True` to "
-            "enable tuning.",
-            UserWarning,
-            stacklevel=3,
-        )
-        return None
-
-    if num_samples < MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING:
-        warnings.warn(
-            f"You have `{num_samples}` samples in the training data and specified "
-            "a tuning configuration. "
-            "We recommend tuning only for datasets with more than "
-            f"{MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING} samples. ",
-            UserWarning,
-            stacklevel=3,
-        )
-
-    return tuning_config.resolve(num_samples=num_samples)
diff --git a/src/tabpfn/misc/_sklearn_compat.py b/src/tabpfn/misc/_sklearn_compat.py
index 48c5d55..a005a0b 100644
--- a/src/tabpfn/misc/_sklearn_compat.py
+++ b/src/tabpfn/misc/_sklearn_compat.py
@@ -1,3 +1,4 @@
+# ruff: noqa
 # mypy: ignore-errors
 # taken from https://github.com/sklearn-compat/sklearn-compat
 """Ease developer experience to support multiple versions of scikit-learn.
diff --git a/src/tabpfn/misc/debug_versions.py b/src/tabpfn/misc/debug_versions.py
index 9ee70e6..2e22df5 100644
--- a/src/tabpfn/misc/debug_versions.py
+++ b/src/tabpfn/misc/debug_versions.py
@@ -1,3 +1,4 @@
+# ruff: noqa
 """This file is taken from PyTorch and modified to work with TabPFN, also
 inspired from sklearn's show_versions function. This collects useful debug
 information that can be used to report issues.
diff --git a/src/tabpfn/model/__init__.py b/src/tabpfn/model/__init__.py
index bd05cb5..e9ecffd 100644
--- a/src/tabpfn/model/__init__.py
+++ b/src/tabpfn/model/__init__.py
@@ -18,6 +18,7 @@ from tabpfn.architectures.base import (
     layer,
     memory,
     mlp,
+    preprocessing,
     transformer,
 )
 
@@ -30,6 +31,7 @@ __all__ = [
     "loading",
     "memory",
     "mlp",
+    "preprocessing",
     "transformer",
 ]
 
diff --git a/src/tabpfn/model/preprocessing.py b/src/tabpfn/model/preprocessing.py
index 37827af..94ba07e 100644
--- a/src/tabpfn/model/preprocessing.py
+++ b/src/tabpfn/model/preprocessing.py
@@ -1,5 +1,5 @@
-"""DEPRECATED: Please import tabpfn.preprocessors instead."""
+"""DEPRECATED: Please import tabpfn.architectures.base.preprocessing instead."""
 
 from __future__ import annotations
 
-from tabpfn.preprocessors import *  # noqa: F403
+from tabpfn.architectures.base.preprocessing import *  # noqa: F403
diff --git a/src/tabpfn/model_loading.py b/src/tabpfn/model_loading.py
index 8c65d31..8c4111b 100644
--- a/src/tabpfn/model_loading.py
+++ b/src/tabpfn/model_loading.py
@@ -12,19 +12,16 @@ import shutil
 import sys
 import tempfile
 import urllib.request
+import urllib.response
 import warnings
-import zipfile
-from copy import deepcopy
-from dataclasses import asdict, dataclass
+from dataclasses import dataclass
 from enum import Enum
-from importlib import import_module
 from pathlib import Path
-from typing import TYPE_CHECKING, Any, Literal, cast, overload
+from typing import TYPE_CHECKING, Literal, cast, overload
 from urllib.error import URLError
 
 import joblib
 import torch
-from tabpfn_common_utils.telemetry import set_model_config
 from torch import nn
 
 from tabpfn.architectures import ARCHITECTURES
@@ -32,35 +29,27 @@ from tabpfn.architectures.base.bar_distribution import (
     BarDistribution,
     FullSupportBarDistribution,
 )
-from tabpfn.constants import ModelVersion
-from tabpfn.inference import InferenceEngine, InferenceEngineCacheKV
-from tabpfn.inference_config import InferenceConfig
+from tabpfn.inference import InferenceEngine
 from tabpfn.settings import settings
 
 if TYPE_CHECKING:
     from sklearn.base import BaseEstimator
 
     from tabpfn import TabPFNClassifier, TabPFNRegressor
-
-if TYPE_CHECKING:
     from tabpfn.architectures.interface import Architecture, ArchitectureConfig
-    from tabpfn.constants import ModelPath
 
 logger = logging.getLogger(__name__)
 
-# Public fallback base URL for model downloads
-FALLBACK_S3_BASE_URL = "https://storage.googleapis.com/tabpfn-v2-model-files/05152025"
-
-# Special string used to identify v2.5 models in model paths.
-V_2_5_IDENTIFIER = "v2.5"
-
 
 class ModelType(str, Enum):  # noqa: D101
-    # TODO: Merge with TaskType in tabpfn.constants.
     CLASSIFIER = "classifier"
     REGRESSOR = "regressor"
 
 
+class ModelVersion(str, Enum):  # noqa: D101
+    V2 = "v2"
+
+
 @dataclass
 class ModelSource:  # noqa: D101
     repo_id: str
@@ -77,12 +66,6 @@ class ModelSource:  # noqa: D101
             "tabpfn-v2-classifier-vutqq28w.ckpt",
             "tabpfn-v2-classifier-znskzxi4.ckpt",
             "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
-            "tabpfn-v2-classifier-finetuned-znskzxi4-tvvss6bp.ckpt",
-            "tabpfn-v2-classifier-finetuned-vutqq28w-boexhu6h.ckpt",
-            "tabpfn-v2-classifier-finetuned-od3j1g5m-4svepuy5.ckpt",
-            "tabpfn-v2-classifier-finetuned-llderlii-oyd7ul21.ckpt",
-            "tabpfn-v2-classifier-finetuned-gn2p4bpt-xp6f0iqb.ckpt",
-            "tabpfn-v2-classifier-v2_default.ckpt",
         ]
         return cls(
             repo_id="Prior-Labs/TabPFN-v2-clf",
@@ -97,7 +80,6 @@ class ModelSource:  # noqa: D101
             "tabpfn-v2-regressor-09gpqh39.ckpt",
             "tabpfn-v2-regressor-2noar4o2.ckpt",
             "tabpfn-v2-regressor-wyl4o83o.ckpt",
-            "tabpfn-v2-regressor-v2_default.ckpt",
         ]
         return cls(
             repo_id="Prior-Labs/TabPFN-v2-reg",
@@ -105,41 +87,11 @@ class ModelSource:  # noqa: D101
             filenames=filenames,
         )
 
-    @classmethod
-    def get_classifier_v2_5(cls) -> ModelSource:  # noqa: D102
-        filenames = [
-            "tabpfn-v2.5-classifier-v2.5_default.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_large-samples.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_real.ckpt",
-            "tabpfn-v2.5-classifier-v2.5_variant.ckpt",
+    def get_fallback_urls(self) -> list[str]:  # noqa: D102
+        return [
+            f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}?download=true"
+            for filename in self.filenames
         ]
-        return cls(
-            repo_id="Prior-Labs/tabpfn_2_5",
-            default_filename="tabpfn-v2.5-classifier-v2.5_default.ckpt",
-            filenames=filenames,
-        )
-
-    @classmethod
-    def get_regressor_v2_5(cls) -> ModelSource:  # noqa: D102
-        filenames = [
-            "tabpfn-v2.5-regressor-v2.5_default.ckpt",
-            "tabpfn-v2.5-regressor-v2.5_low-skew.ckpt",
-            "tabpfn-v2.5-regressor-v2.5_quantiles.ckpt",
-            "tabpfn-v2.5-regressor-v2.5_real-variant.ckpt",
-            "tabpfn-v2.5-regressor-v2.5_real.ckpt",
-            "tabpfn-v2.5-regressor-v2.5_small-samples.ckpt",
-            "tabpfn-v2.5-regressor-v2.5_variant.ckpt",
-        ]
-        return cls(
-            repo_id="Prior-Labs/tabpfn_2_5",
-            default_filename="tabpfn-v2.5-regressor-v2.5_default.ckpt",
-            filenames=filenames,
-        )
 
 
 def _get_model_source(version: ModelVersion, model_type: ModelType) -> ModelSource:
@@ -148,11 +100,6 @@ def _get_model_source(version: ModelVersion, model_type: ModelType) -> ModelSour
             return ModelSource.get_classifier_v2()
         if model_type == ModelType.REGRESSOR:
             return ModelSource.get_regressor_v2()
-    elif version == ModelVersion.V2_5:
-        if model_type == ModelType.CLASSIFIER:
-            return ModelSource.get_classifier_v2_5()
-        if model_type == ModelType.REGRESSOR:
-            return ModelSource.get_regressor_v2_5()
 
     raise ValueError(
         f"Unsupported version/model combination: {version.value}/{model_type.value}",
@@ -176,13 +123,7 @@ def _try_huggingface_downloads(
     """
     """Try to download models and config using the HuggingFace Hub API."""
     try:
-        from huggingface_hub import hf_hub_download  # noqa: PLC0415
-
-        # Import specific exceptions for better error handling
-        from huggingface_hub.utils import (  # noqa: PLC0415
-            GatedRepoError,
-            HfHubHTTPError,
-        )
+        from huggingface_hub import hf_hub_download
     except ImportError as e:
         raise ImportError(
             "Please install huggingface_hub: pip install huggingface-hub",
@@ -224,8 +165,6 @@ def _try_huggingface_downloads(
 
             # Download config.json only to increment the download counter. We do not
             # actually use this file so it is removed immediately after download.
-            # Note that we also handle model caching ourselves, so we don't double
-            # count, even with removing the config.json afterwards.
             try:
                 config_local_path = hf_hub_download(
                     repo_id=source.repo_id,
@@ -238,33 +177,8 @@ def _try_huggingface_downloads(
                 # Continue even if config.json download fails
 
             logger.info(f"Successfully downloaded to {base_path}")
-
-        except (GatedRepoError, HfHubHTTPError) as e:
-            # Check if this is an authentication/gating error
-            is_auth_error = False
-            if isinstance(e, GatedRepoError) or (
-                isinstance(e, HfHubHTTPError) and e.response.status_code in (401, 403)
-            ):
-                is_auth_error = True
-
-            if is_auth_error:
-                auth_message = (
-                    f"Authentication error downloading from '{source.repo_id}'.\n"
-                    "This model is gated and requires you to accept its terms.\n\n"
-                    "Please follow these steps:\n"
-                    f"1. Visit https://huggingface.co/{source.repo_id} in your "
-                    f"browser and"
-                    f" accept the terms of use.\n"
-                    "2. Log in to your Hugging Face account via"
-                    " the command line by running:\n"
-                    "   hf auth login\n"
-                    "(Alternatively, you can set the HF_TOKEN environment variable"
-                    " with a read token).\n\n"
-                    "For detailed instructions, see "
-                    "https://docs.priorlabs.ai/how-to-access-gated-models"
-                )
-                raise RuntimeError(auth_message)  # noqa: B904
-            raise e
+        except Exception as e:
+            raise Exception("HuggingFace download failed!") from e
 
 
 def _try_direct_downloads(
@@ -284,48 +198,44 @@ def _try_direct_downloads(
         if filename not in source.filenames:
             source.filenames.append(filename)
 
-    url_pairs = [
-        (
-            f"https://huggingface.co/{source.repo_id}/resolve/main/{filename}?download=true",
-            f"https://huggingface.co/{source.repo_id}/resolve/main/config.json?download=true",
-        ),
-        (f"{FALLBACK_S3_BASE_URL}/{filename}", f"{FALLBACK_S3_BASE_URL}/config.json"),
-    ]
+    model_url = (
+        f"https://huggingface.co/{source.repo_id}/resolve/main/{filename}?download=true"
+    )
+    config_url = f"https://huggingface.co/{source.repo_id}/resolve/main/config.json?download=true"
 
-    last_error: Exception | None = None
+    # Create parent directory if it doesn't exist
     base_path.parent.mkdir(parents=True, exist_ok=True)
 
-    for model_url, config_url in url_pairs:
-        logger.info(f"Attempting download from {model_url}")
-        try:
-            with urllib.request.urlopen(model_url, timeout=180) as response:  # noqa: S310
-                if response.status != 200:
-                    raise URLError(
-                        f"HTTP {response.status} when downloading from {model_url}",
-                    )
-                base_path.write_bytes(response.read())
-
-            config_path = base_path.parent / "config.json"
-            try:
-                with urllib.request.urlopen(config_url, timeout=180) as response:  # noqa: S310
-                    if response.status == 200:
-                        config_path.write_bytes(response.read())
-            except Exception:  # noqa: BLE001
-                logger.warning("Failed to download config.json!")
+    logger.info(f"Attempting download from {model_url}")
 
-            logger.info(f"Successfully downloaded to {base_path}")
-            return
-        except Exception as e:  # noqa: BLE001
-            last_error = e
-            logger.warning(f"Failed download from {model_url}: {e!s}")
+    try:
+        # Download model checkpoint
+        with urllib.request.urlopen(model_url) as response:  # noqa: S310
+            if response.status != 200:
+                raise URLError(
+                    f"HTTP {response.status} when downloading from {model_url}",
+                )
+            base_path.write_bytes(response.read())
+
+        # Try to download config.json
+        config_path = base_path.parent / "config.json"
+        try:
+            with urllib.request.urlopen(config_url) as response:  # noqa: S310
+                if response.status == 200:
+                    config_path.write_bytes(response.read())
+        except Exception:  # noqa: BLE001
+            logger.warning("Failed to download config.json!")
+            # Continue even if config.json download fails
 
-    raise Exception("Direct download failed!") from last_error
+        logger.info(f"Successfully downloaded to {base_path}")
+    except Exception as e:
+        raise Exception("Direct download failed!") from e
 
 
 def download_model(
     to: Path,
     *,
-    version: ModelVersion,
+    version: Literal["v2"],
     which: Literal["classifier", "regressor"],
     model_name: str | None = None,
 ) -> Literal["ok"] | list[Exception]:
@@ -344,7 +254,7 @@ def download_model(
     errors: list[Exception] = []
 
     try:
-        model_source = _get_model_source(version, ModelType(which))
+        model_source = _get_model_source(ModelVersion(version), ModelType(which))
     except ValueError as e:
         return [e]
 
@@ -352,58 +262,42 @@ def download_model(
         _try_huggingface_downloads(to, model_source, model_name, suppress_warnings=True)
         return "ok"
     except Exception as e:  # noqa: BLE001
-        logger.warning("HuggingFace download failed.")
+        logger.warning(f"HuggingFace downloads failed: {e!s}")
         errors.append(e)
 
-    # For Version 2.5 we require gating, which we don't have in place for direct
-    # downloads.
-    if version == ModelVersion.V2:
-        try:
-            _try_direct_downloads(to, model_source, model_name)
-            return "ok"
-        except Exception as e:  # noqa: BLE001
-            logger.warning(f"Direct URL downloads failed: {e!s}")
-            errors.append(e)
-    else:
-        logger.warning(
-            "For commercial usage, we provide alternative download options for v2.5, "
-            "please reach out to us at sales@priorlabs.ai."
-        )
+    try:
+        _try_direct_downloads(to, model_source, model_name)
+        return "ok"
+    except Exception as e:  # noqa: BLE001
+        logger.warning(f"Direct URL downloads failed: {e!s}")
+        errors.append(e)
 
     return errors
 
 
 def download_all_models(to: Path) -> None:
-    """Download all available classifier and regressor models into a local directory."""
+    """Download all v2 classifier and regressor models into a local directory."""
     to.mkdir(parents=True, exist_ok=True)
-    for model_version, model_source, model_type in [
-        (ModelVersion.V2, ModelSource.get_classifier_v2(), "classifier"),
-        (ModelVersion.V2, ModelSource.get_regressor_v2(), "regressor"),
-        (ModelVersion.V2_5, ModelSource.get_classifier_v2_5(), "classifier"),
-        (ModelVersion.V2_5, ModelSource.get_regressor_v2_5(), "regressor"),
+    for model_source, model_type in [
+        (ModelSource.get_classifier_v2(), "classifier"),
+        (ModelSource.get_regressor_v2(), "regressor"),
     ]:
         for ckpt_name in model_source.filenames:
             download_model(
                 to=to / ckpt_name,
-                version=model_version,
+                version="v2",
                 which=cast("Literal['classifier', 'regressor']", model_type),
                 model_name=ckpt_name,
             )
 
 
-def get_cache_dir() -> Path:  # noqa: PLR0911
-    """Get the cache directory for TabPFN models, as appropriate for the platform."""
-    if settings.tabpfn.model_cache_dir is not None:
-        return settings.tabpfn.model_cache_dir
-
-    platform = sys.platform
-    appname = "tabpfn"
+def _user_cache_dir(platform: str, appname: str = "tabpfn") -> Path:
     use_instead_path = (Path.cwd() / ".tabpfn_models").resolve()
 
+    # https://docs.python.org/3/library/sys.html#sys.platform
     if platform == "win32":
-        # Do something similar to platformdirs, but very simplified:
+        # Honestly, I don't want to do what `platformdirs` does:
         # https://github.com/tox-dev/platformdirs/blob/b769439b2a3b70769a93905944a71b3e63ef4823/src/platformdirs/windows.py#L252-L265
-        # Unclear how well this works.
         APPDATA_PATH = os.environ.get("APPDATA", "")
         if APPDATA_PATH.strip() != "":
             return Path(APPDATA_PATH) / appname
@@ -447,60 +341,49 @@ def get_cache_dir() -> Path:  # noqa: PLR0911
 
 @overload
 def load_model_criterion_config(
-    model_path: ModelPath | list[ModelPath] | None,
+    model_path: str | Path | None,
     *,
     check_bar_distribution_criterion: Literal[False],
     cache_trainset_representation: bool,
-    version: Literal["v2", "v2.5"],
+    version: Literal["v2"],
     which: Literal["classifier"],
-    download_if_not_exists: bool,
+    download: bool,
 ) -> tuple[
-    list[Architecture],
+    Architecture,
     nn.BCEWithLogitsLoss | nn.CrossEntropyLoss,
-    list[ArchitectureConfig],
-    InferenceConfig,
+    ArchitectureConfig,
 ]: ...
 
 
 @overload
 def load_model_criterion_config(
-    model_path: ModelPath | list[ModelPath] | None,
+    model_path: str | Path | None,
     *,
     check_bar_distribution_criterion: Literal[True],
     cache_trainset_representation: bool,
-    version: Literal["v2", "v2.5"],
+    version: Literal["v2"],
     which: Literal["regressor"],
-    download_if_not_exists: bool,
-) -> tuple[
-    list[Architecture],
-    FullSupportBarDistribution,
-    list[ArchitectureConfig],
-    InferenceConfig,
-]: ...
+    download: bool,
+) -> tuple[Architecture, FullSupportBarDistribution, ArchitectureConfig]: ...
 
 
 def load_model_criterion_config(
-    model_path: ModelPath | list[ModelPath] | None,
+    model_path: None | str | Path,
     *,
     check_bar_distribution_criterion: bool,
     cache_trainset_representation: bool,
     which: Literal["regressor", "classifier"],
-    version: Literal["v2", "v2.5"] = "v2",
-    download_if_not_exists: bool,
+    version: Literal["v2"] = "v2",
+    download: bool,
 ) -> tuple[
-    list[Architecture],
+    Architecture,
     nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
-    list[ArchitectureConfig],
-    InferenceConfig,
+    ArchitectureConfig,
 ]:
-    """Load the model(s), criterion(s), and config(s) from the given path.
-
-    If multiple model paths are provided, then all models must use the same criterion
-    and inference config.
+    """Load the model, criterion, and config from the given path.
 
     Args:
-        model_path: The path to the model, or list of paths if multiple models should be
-            loaded.
+        model_path: The path to the model.
         check_bar_distribution_criterion:
             Whether to check if the criterion
             is a FullSupportBarDistribution, which is the expected criterion
@@ -509,152 +392,58 @@ def load_model_criterion_config(
             Whether the model should know to cache the trainset representation.
         which: Whether the model is a regressor or classifier.
         version: The version of the model.
-        download_if_not_exists: Whether to download the model if it doesn't exist.
+        download: Whether to download the model if it doesn't exist.
 
     Returns:
-        list of models, the criterion, list of architecture configs, the inference
-        config
+        The model, criterion, and config.
     """
-    model_version = ModelVersion(version)
-    (resolved_model_paths, resolved_model_dirs, resolved_model_names, which) = (
-        resolve_model_path(
-            model_path=model_path,
-            which=which,
-            version=model_version.value,
-        )
+    (model_path, model_dir, model_name, which) = resolve_model_path(
+        model_path, which, version
     )
 
-    # Anonymously track the model config for usage telemetry
-    _log_model_config(resolved_model_paths, which, model_version)
-
-    for folder in resolved_model_dirs:
-        folder.mkdir(parents=True, exist_ok=True)
-
-    loaded_models = []
-    criterions = []
-    architecture_configs = []
-    inference_configs = []
-
-    for i, path in enumerate(resolved_model_paths):
-        if not path.exists():
-            if not download_if_not_exists:
-                raise ValueError(
-                    f"Model path does not exist and downloading is disabled"
-                    f"\nmodel path: {path}",
-                )
-
-            logger.info(f"Downloading model to {path}.")
-            res = download_model(
-                path,
-                version=model_version,
-                which=cast("Literal['classifier', 'regressor']", which),
-                model_name=resolved_model_names[i],
-            )
-            if res != "ok":
-                # Later: Add improved error handling here, reenabling
-                #  the old offline download (only raise when Gating)
-                raise res[0]
-
-        loaded_model, criterion, architecture_config, inference_config = load_model(
-            path=path,
-            cache_trainset_representation=cache_trainset_representation,
-        )
-        if check_bar_distribution_criterion and not isinstance(
-            criterion,
-            FullSupportBarDistribution,
-        ):
-            raise ValueError(
-                f"The model loaded, '{path}', was expected to have a"
-                " FullSupportBarDistribution criterion, but instead "
-                f" had a {type(criterion).__name__} criterion.",
-            )
-        loaded_models.append(loaded_model)
-        criterions.append(criterion)
-        architecture_configs.append(architecture_config)
-        inference_configs.append(inference_config)
-
-    first_criterion = criterions[0]
-    if isinstance(first_criterion, FullSupportBarDistribution):
-        for criterion in criterions[1:]:
-            if not first_criterion.has_equal_borders(criterion):
-                raise ValueError(
-                    f"The criterions {first_criterion} and {criterion} are different. "
-                    "This is not supported in the current implementation"
-                )
-
-    first_inference_config = inference_configs[0]
-    for inference_config in inference_configs[1:]:
-        if inference_config != first_inference_config:
+    model_dir.mkdir(parents=True, exist_ok=True)
+    if not model_path.exists():
+        if not download:
             raise ValueError(
-                f"Config 1: {first_inference_config}\n"
-                f"Config 2: {inference_config}\n"
-                "Inference configs for different models are different, which is not "
-                "supported. See above."
+                f"Model path does not exist and downloading is disabled"
+                f"\nmodel path: {model_path}",
             )
 
-    return loaded_models, first_criterion, architecture_configs, first_inference_config
-
-
-def _resolve_model_version(model_path: ModelPath | None) -> ModelVersion:
-    if model_path is None:
-        return settings.tabpfn.model_version
-    if V_2_5_IDENTIFIER in Path(model_path).name:
-        return ModelVersion.V2_5
-    return ModelVersion.V2
-
-
-def _log_model_config(
-    model_paths: list[Path],
-    which: Literal["classifier", "regressor"],
-    version: ModelVersion,
-) -> None:
-    """Set the model config (model_path and model_version) for anonymous
-    usage telemetry.
-
-    Args:
-        model_paths: The path(s) to the model.
-        which: The type of model ('classifier' or 'regressor').
-        version: The model version (currently only 'v2' or 'v2.5').
-    """
-    if len(model_paths) != 1:
-        return
-
-    model_type = ModelType(which)
-    model_source = _get_model_source(version, model_type)
-
-    path: Path = model_paths[0]
-    # Check to avoid that we pass in arbitrary paths containing e.g. PII
-    # Ensure we whitelist model names so that no PII can be released.
-    if path.name in model_source.filenames:
-        set_model_config(path.name, version.value)
-    else:
-        set_model_config("OTHER", version.value)
-
-
-def resolve_model_version(
-    model_path: ModelPath | list[ModelPath] | None,
-) -> ModelVersion:
-    """Resolve the model version from the model path."""
-    if isinstance(model_path, list):
-        if len(model_path) == 0:
-            return _resolve_model_version(None)
-        resolved_model_versions = [_resolve_model_version(p) for p in model_path]
-        if len(set(resolved_model_versions)) > 1:
-            raise ValueError("All model paths must have the same version.")
-        return resolved_model_versions[0]
-    return _resolve_model_version(model_path)
+        logger.info(f"Downloading model to {model_path}.")
+        res = download_model(
+            model_path,
+            version=version,
+            which=cast("Literal['classifier', 'regressor']", which),
+            model_name=model_name,
+        )
+        if res != "ok":
+            repo_type = "clf" if which == "classifier" else "reg"
+            raise RuntimeError(
+                f"Failed to download model to {model_path}!\n\n"
+                f"For offline usage, please download the model manually from:\n"
+                f"https://huggingface.co/Prior-Labs/TabPFN-v2-{repo_type}/resolve/main/{model_name}\n\n"
+                f"Then place it at: {model_path}",
+            ) from res[0]
+
+    loaded_model, criterion, config = load_model(path=model_path)
+    loaded_model.cache_trainset_representation = cache_trainset_representation
+    if check_bar_distribution_criterion and not isinstance(
+        criterion,
+        FullSupportBarDistribution,
+    ):
+        raise ValueError(
+            f"The model loaded, '{model_path}', was expected to have a"
+            " FullSupportBarDistribution criterion, but instead "
+            f" had a {type(criterion).__name__} criterion.",
+        )
+    return loaded_model, criterion, config
 
 
 def resolve_model_path(
-    model_path: ModelPath | list[ModelPath] | None,
+    model_path: None | str | Path,
     which: Literal["regressor", "classifier"],
-    version: Literal["v2", "v2.5"] = "v2.5",
-) -> tuple[
-    list[Path],
-    list[Path],
-    list[str],
-    Literal["regressor", "classifier"],
-]:
+    version: Literal["v2"] = "v2",
+) -> tuple[Path, Path, str, str]:
     """Resolves the model path, using the official default model if no path is provided.
 
     Args:
@@ -665,27 +454,32 @@ def resolve_model_path(
         version: The model version (currently only 'v2').
 
     Returns:
-        A tuple containing lists of resolved model Path(s),
-        the parent directory Path(s), the model's filename(s), and model type Literal.
+        A tuple containing the resolved model Path, the parent directory Path,
+        the model's filename, and the model type.
     """
     if model_path is None:
         # Get the source information to find the official default model filename.
         model_source = _get_model_source(ModelVersion(version), ModelType(which))
-        resolved_model_names = [model_source.default_filename]
+        model_name = model_source.default_filename
 
         # Determine the cache directory for storing models.
-        resolved_model_dirs = [get_cache_dir()]
-        resolved_model_paths = [resolved_model_dirs[0] / resolved_model_names[0]]
-    elif isinstance(model_path, (str, Path)):
-        resolved_model_paths = [Path(model_path)]
-        resolved_model_dirs = [resolved_model_paths[0].parent]
-        resolved_model_names = [resolved_model_paths[0].name]
+        if settings.tabpfn.model_cache_dir is not None:
+            model_dir = settings.tabpfn.model_cache_dir
+        else:
+            model_dir = _user_cache_dir(platform=sys.platform, appname="tabpfn")
+
+        # Construct the full path to the default model.
+        model_path = model_dir / model_name
     else:
-        resolved_model_paths = [Path(p) for p in model_path]
-        resolved_model_dirs = [p.parent for p in resolved_model_paths]
-        resolved_model_names = [p.name for p in resolved_model_paths]
+        # If a path is provided, simply parse it.
+        if not isinstance(model_path, (str, Path)):
+            raise ValueError(f"Invalid model_path: {model_path}")
 
-    return resolved_model_paths, resolved_model_dirs, resolved_model_names, which
+        model_path = Path(model_path)
+        model_dir = model_path.parent
+        model_name = model_path.name
+
+    return model_path, model_dir, model_name, which
 
 
 def get_loss_criterion(
@@ -718,19 +512,15 @@ def get_loss_criterion(
 def load_model(
     *,
     path: Path,
-    cache_trainset_representation: bool = True,
 ) -> tuple[
     Architecture,
     nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
     ArchitectureConfig,
-    InferenceConfig,
 ]:
     """Loads a model from a given path. Only for inference.
 
     Args:
         path: Path to the checkpoint
-        cache_trainset_representation: If True, the model will cache the
-            trainset representation. Forwarded to get_architecture.
     """
     # Catch the `FutureWarning` that torch raises. This should be dealt with!
     # The warning is raised due to `torch.load`, which advises against ckpt
@@ -739,7 +529,7 @@ def load_model(
     # `True`, dissallowing loading of arbitrary objects.
     with warnings.catch_warnings():
         warnings.simplefilter("ignore", category=FutureWarning)
-        checkpoint: dict = torch.load(path, map_location="cpu", weights_only=None)
+        checkpoint = torch.load(path, map_location="cpu", weights_only=None)
 
     try:
         architecture_name = checkpoint["architecture_name"]
@@ -747,14 +537,14 @@ def load_model(
         architecture_name = "base"
     architecture = ARCHITECTURES[architecture_name]
     state_dict = checkpoint["state_dict"]
-    model_config, unused_model_config = architecture.parse_config(checkpoint["config"])
+    config, unused_config = architecture.parse_config(checkpoint["config"])
     logger.debug(
         "Keys in config that were not parsed by architecture config: "
-        f"{', '.join(unused_model_config.keys())}"
+        f"{', '.join(unused_config.keys())}"
     )
 
     criterion_state_keys = [k for k in state_dict if "criterion." in k]
-    loss_criterion = get_loss_criterion(model_config)
+    loss_criterion = get_loss_criterion(config)
     if isinstance(loss_criterion, FullSupportBarDistribution):
         # Remove from state dict
         criterion_state = {
@@ -765,45 +555,14 @@ def load_model(
         assert len(criterion_state_keys) == 0, criterion_state_keys
 
     model = architecture.get_architecture(
-        model_config,
-        n_out=get_n_out(model_config, loss_criterion),
-        cache_trainset_representation=cache_trainset_representation,
+        config,
+        n_out=get_n_out(config, loss_criterion),
+        cache_trainset_representation=True,
     )
     model.load_state_dict(state_dict)
     model.eval()
 
-    inference_config = _get_inference_config_from_checkpoint(checkpoint, loss_criterion)
-
-    return model, loss_criterion, model_config, inference_config
-
-
-def _get_inference_config_from_checkpoint(
-    checkpoint: dict[str, Any],
-    criterion: nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
-) -> InferenceConfig:
-    """Return the config in the checkpoint, or an appropriate default config.
-
-    We only started storing the inference config in the checkpoint after the v2.5
-    release. Thus, if there is no config in the checkpoint, try to guess between v2 and
-    v2.5 and get the correct config.
-    """
-    # This is how we tell the checkpoints apart:
-    #     v2: "architecture_name" not present, as added after the v2 release
-    #   v2.5: "architecture_name" present, but "inference_config" not present
-    #  >v2.5: "inference_config" present, so don't need to guess a default config
-    if inference_config := checkpoint.get("inference_config"):
-        return InferenceConfig(**inference_config)
-    if "architecture_name" not in checkpoint:
-        model_version = ModelVersion.V2
-    else:
-        model_version = ModelVersion.V2_5
-
-    if isinstance(criterion, FullSupportBarDistribution):
-        task_type = "regression"
-    else:
-        task_type = "multiclass"
-
-    return InferenceConfig.get_default(task_type, model_version)
+    return model, loss_criterion, config
 
 
 def get_n_out(
@@ -824,8 +583,7 @@ def get_n_out(
 
 
 def save_tabpfn_model(
-    model: TabPFNRegressor | TabPFNClassifier,
-    save_path: Path | str | list[Path | str],
+    model: TabPFNRegressor | TabPFNClassifier, save_path: Path | str
 ) -> None:
     """Save the underlying TabPFN foundation model to ``save_path``.
 
@@ -840,50 +598,27 @@ def save_tabpfn_model(
         save_path:
             Path to save the checkpoint to.
     """
-    if len(model.models_) > 1 and (
-        not isinstance(save_path, list) or len(save_path) != len(model.models_)
-    ):
-        raise ValueError(
-            f"Your TabPFN estimator has multiple internal models ({len(model.models_)})"
-            f", so you must provide a list of {len(model.models_)} save paths."
-        )
-
-    models = model.models_
+    # Get model state dict
+    model_state = model.model_.state_dict()
 
-    znorm_space_bardist = None
-    if (
-        hasattr(model, "znorm_space_bardist_")
-        and model.znorm_space_bardist_ is not None  # type: ignore
-    ):
-        znorm_space_bardist = model.znorm_space_bardist_  # type: ignore
-
-    configs = model.configs_
-    save_paths = save_path if isinstance(save_path, list) else [save_path]
-
-    for ens_model, config, path in zip(
-        models,
-        configs,
-        save_paths,
-    ):
-        model_state = ens_model.state_dict()
-
-        if znorm_space_bardist is not None:
-            bardist_state = {
-                f"criterion.{k}": v for k, v in znorm_space_bardist.state_dict().items()
-            }
-            state_dict = {**model_state, **bardist_state}
-        else:
-            state_dict = model_state
+    # Get bardist state dict and prefix with 'criterion.'
+    if hasattr(model, "bardist_") and model.bardist_ is not None:
+        bardist_state = {
+            f"criterion.{k}": v for k, v in model.bardist_.state_dict().items()
+        }
+        # Combine model and bardist states
+        state_dict = {**model_state, **bardist_state}
+    else:
+        state_dict = model_state
 
-        checkpoint = {"state_dict": state_dict, "config": asdict(config)}
+    # Create checkpoint with correct structure
+    checkpoint = {"state_dict": state_dict, "config": model.config_}
 
-        torch.save(checkpoint, path)
+    # Save the checkpoint
+    torch.save(checkpoint, save_path)
 
 
-def save_fitted_tabpfn_model(
-    estimator: BaseEstimator,
-    path: Path | str,
-) -> None:
+def save_fitted_tabpfn_model(estimator: BaseEstimator, path: Path | str) -> None:
     """Persist a fitted TabPFN estimator to ``path``.
 
     This stores the initialization parameters and the fitted state, but crucially
@@ -897,7 +632,7 @@ def save_fitted_tabpfn_model(
         raise ValueError("Path must end with .tabpfn_fit")
 
     # Attributes that are handled separately or should not be saved.
-    blacklist = {"models_", "executor_", "configs_", "devices_"}
+    blacklist = {"model_", "executor_", "config_", "device_"}
 
     with tempfile.TemporaryDirectory() as tmpdir:
         tmp = Path(tmpdir)
@@ -917,18 +652,10 @@ def save_fitted_tabpfn_model(
             for key, value in vars(estimator).items()
             if key.endswith("_") and key not in blacklist
         }
-        # move all tensors to "cpu" before saving, so if fitted & saved on cuda-device
-        # and loading on cpu-device does not throw
-        # "RuntimeError: Attempting to deserialize object on a CUDA device..."
-        fitted_attrs = {
-            k: v.to("cpu") if isinstance(v, (torch.nn.Module, torch.Tensor)) else v
-            for k, v in fitted_attrs.items()
-        }
-
         joblib.dump(fitted_attrs, tmp / "fitted_attrs.joblib")
 
         # 3. Save the InferenceEngine state without the model weights
-        estimator.executor_.save_state_except_model_weights(
+        estimator.executor_.save_state_expect_model_weights(
             tmp / "executor_state.joblib"
         )
 
@@ -938,6 +665,8 @@ def save_fitted_tabpfn_model(
 
 
 def _extract_archive(path: Path, tmp: Path) -> None:
+    import zipfile
+
     with zipfile.ZipFile(path, "r") as archive:
         for member in archive.namelist():
             member_path = (tmp / member).resolve()
@@ -950,12 +679,8 @@ def load_fitted_tabpfn_model(
     path: Path | str, *, device: str | torch.device = "cpu"
 ) -> BaseEstimator:
     """Load a fitted TabPFN estimator saved with ``save_fitted_tabpfn_model``."""
-    # This is safe because torch.device(torch.device(...)) is a noop.
-    device = torch.device(device)
-    # In older versions of PyTorch, some torch.cuda functions fail if the device has no
-    # index. 0 is implicit if no index is specified, so add it.
-    if device == torch.device("cuda"):
-        device = torch.device("cuda:0")
+    from copy import deepcopy
+    from importlib import import_module
 
     path = Path(path)
     with tempfile.TemporaryDirectory() as tmpdir:
@@ -984,7 +709,7 @@ def load_fitted_tabpfn_model(
             raise TypeError(f"Unknown estimator class '{saved_cls_name}'")
 
         est = cls(**params)
-        # This is critical: it loads the base model weights into `est.models_`
+        # This is critical: it loads the base model weights into `est.model_`
         est._initialize_model_variables()
 
         # 2. Restore all other fitted attributes
@@ -996,23 +721,24 @@ def load_fitted_tabpfn_model(
         est.executor_ = InferenceEngine.load_state(tmp / "executor_state.joblib")
 
         # 4. Re-link the foundation model with the loaded engine
-        if est.executor_.models is None:
-            if isinstance(est.executor_, InferenceEngineCacheKV):  # type: ignore
-                est.executor_.models = [
-                    deepcopy(est.models_[config._model_index])
-                    for config in est.executor_.ensemble_configs  # type: ignore
-                ]
-            else:
-                est.executor_.models = [deepcopy(model) for model in est.models_]
+        if hasattr(est.executor_, "model") and est.executor_.model is None:
+            est.executor_.model = est.model_
+
+        if hasattr(est.executor_, "models") and est.executor_.models is None:
+            est.executor_.models = [
+                deepcopy(est.model_) for _ in range(len(est.executor_.ensemble_configs))
+            ]
 
         # 5. Move all torch components to the target device
-        est.devices_ = (torch.device(device),)
+        est.device_ = torch.device(device)
+        if hasattr(est.executor_, "model") and est.executor_.model is not None:
+            est.executor_.model.to(est.device_)
         if hasattr(est.executor_, "models"):
-            est.executor_.models = [m.to(device) for m in est.executor_.models]
+            est.executor_.models = [m.to(est.device_) for m in est.executor_.models]
 
         # Restore other potential torch objects from fitted_attrs
         for key, value in vars(est).items():
             if key.endswith("_") and hasattr(value, "to"):
-                setattr(est, key, value.to(device))
+                setattr(est, key, value.to(est.device_))
 
         return est
diff --git a/src/tabpfn/parallel_execute.py b/src/tabpfn/parallel_execute.py
deleted file mode 100644
index f3203de..0000000
--- a/src/tabpfn/parallel_execute.py
+++ /dev/null
@@ -1,114 +0,0 @@
-"""Parallel evaluation of a set of functions across multiple PyTorch devices."""
-
-from __future__ import annotations
-
-import queue
-from collections.abc import Generator, Iterable, Sequence
-from multiprocessing.pool import ThreadPool
-from typing import Generic, Protocol, TypeVar
-
-import torch
-
-R_co = TypeVar("R_co", covariant=True)
-
-
-class ParallelFunction(Protocol, Generic[R_co]):
-    """Interface that functions submitted to `parallel_execute()` should implement."""
-
-    def __call__(self, *, device: torch.device, is_parallel: bool) -> R_co:
-        """Execute the function.
-
-        If using CUDA, `parallel_execute()` will set the current stream, and this
-        function should not change it.
-
-        Args:
-            device: PyTorch device that all computation should be performed on.
-            is_parallel: Indicates whether this function is being executed in parallel
-                with other functions. If True, then the function should take care to
-                copy any state shared with other functions before mutating it. For
-                example, any nn.Modules should be deep copied before moving them to
-                `device`. If False, then copying can be avoided to reduce overhead.
-
-        Returns:
-            Any desired value. Any Tensors in the returned value should be on `device`.
-        """
-        ...
-
-
-def parallel_execute(
-    devices: Sequence[torch.device],
-    functions: Iterable[ParallelFunction[R_co]],
-) -> Generator[R_co]:
-    """Evaluate the given functions in parallel across `devices`.
-
-    The function evaluations are parallelised using Python threads, so this will only
-    result in a speed-up if the functions do not hold the global interpreter lock. It
-    works well for functions that spend most of their time executing GPU kernels.
-
-    If only one device is provided, then the functions are executed in the current
-    thread to reduce overhead.
-
-    Args:
-        devices: The devices to use for evaluation.
-        functions: The functions to evaluate following the `ParallelFunction` protocol.
-
-    Returns:
-        A generator consisting of the return values of the functions, in the same order
-        as `functions`.
-    """
-    if len(devices) == 1:
-        # If we only have one device then just use the current thread to avoid overhead.
-        yield from _execute_in_current_thread(devices[0], functions)
-    else:
-        yield from _execute_with_multithreading(devices, functions)
-
-
-def _execute_in_current_thread(
-    device: torch.device, functions: Iterable[ParallelFunction[R_co]]
-) -> Generator[R_co]:
-    for function in functions:
-        yield function(device=device, is_parallel=False)
-
-
-def _execute_with_multithreading(
-    devices: Sequence[torch.device],
-    functions: Iterable[ParallelFunction[R_co]],
-) -> Generator[R_co]:
-    free_devices: queue.Queue[int] = queue.Queue(maxsize=len(devices))
-    for device_index, _ in enumerate(devices):
-        free_devices.put(device_index, block=False)
-
-    with ThreadPool(processes=len(devices)) as pool:
-        async_results = [
-            pool.apply_async(_execute_function_in_thread, (devices, free_devices, func))
-            for func in functions
-        ]
-        for async_result in async_results:
-            yield async_result.get()
-
-
-def _execute_function_in_thread(
-    all_devices: Sequence[torch.device],
-    free_devices: queue.Queue[int],
-    function: ParallelFunction[R_co],
-) -> R_co:
-    device_index = free_devices.get(block=True)
-    try:
-        device = all_devices[device_index]
-        if device.type == "cuda":
-            # We use a separate stream per thread so that threads can execute kernels in
-            # parallel.
-            stream = torch.cuda.Stream(device)
-            with torch.cuda.stream(stream), torch.cuda.device(device):
-                output = function(device=device, is_parallel=True)
-                # The returned output will be consumed on a different CUDA stream, hence
-                # we synchronize before returning so that the output is ready for the
-                # consumer. It would be more efficient for the consumer to wait, so this
-                # thread can start with the next function, but this approach is simpler.
-                stream.synchronize()
-                return output
-        # Theoretically it is possible to parallelise over classes of device other than
-        # GPUs, but mainly this is useful for unit testing with multiple CPU devices.
-        return function(device=device, is_parallel=True)
-    finally:
-        free_devices.put(device_index)
diff --git a/src/tabpfn/preprocessing.py b/src/tabpfn/preprocessing.py
index 061ec16..7349167 100644
--- a/src/tabpfn/preprocessing.py
+++ b/src/tabpfn/preprocessing.py
@@ -6,27 +6,20 @@ different members.
 
 from __future__ import annotations
 
-import warnings
-from collections.abc import Callable, Iterable, Iterator, Sequence
-from dataclasses import dataclass, field
+from collections.abc import Iterable, Iterator, Sequence
+from dataclasses import dataclass
 from functools import partial
 from itertools import chain, product, repeat
 from typing import TYPE_CHECKING, Literal, TypeVar
 from typing_extensions import override
 
-import joblib
 import numpy as np
 import torch
+from sklearn.utils.validation import joblib
 from torch.utils.data import Dataset
 
 from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
-from tabpfn.constants import (
-    CLASS_SHUFFLE_OVERESTIMATE_FACTOR,
-    MAXIMUM_FEATURE_SHIFT,
-    PARALLEL_MODE_TO_RETURN_AS,
-    SUPPORTS_RETURN_AS,
-)
-from tabpfn.preprocessors import (
+from tabpfn.architectures.base.preprocessing import (
     AddFingerprintFeaturesStep,
     DifferentiableZNormStep,
     EncodeCategoricalFeaturesStep,
@@ -37,6 +30,12 @@ from tabpfn.preprocessors import (
     SequentialFeatureTransformer,
     ShuffleFeaturesStep,
 )
+from tabpfn.constants import (
+    CLASS_SHUFFLE_OVERESTIMATE_FACTOR,
+    MAXIMUM_FEATURE_SHIFT,
+    PARALLEL_MODE_TO_RETURN_AS,
+    SUPPORTS_RETURN_AS,
+)
 from tabpfn.utils import infer_random_state
 
 if TYPE_CHECKING:
@@ -49,10 +48,7 @@ T = TypeVar("T")
 
 
 def balance(x: Iterable[T], n: int) -> list[T]:
-    """Take a list of elements and make a new list where each appears `n` times.
-
-    E.g. balance([1, 2, 3], 2) -> [1, 1, 2, 2, 3, 3]
-    """
+    """Take a list of elements and make a new list where each appears `n` times."""
     return list(chain.from_iterable(repeat(elem, n) for elem in x))
 
 
@@ -75,33 +71,7 @@ class ClassifierDatasetConfig(BaseDatasetConfig):
 class RegressorDatasetConfig(BaseDatasetConfig):
     """Regression Dataset + Model Configuration class."""
 
-    znorm_space_bardist_: FullSupportBarDistribution | None = field(default=None)
-
-    @property
-    def bardist_(self) -> FullSupportBarDistribution:
-        """DEPRECATED: Accessing `bardist_` is deprecated.
-        Use `znorm_space_bardist_` instead.
-        """
-        warnings.warn(
-            "`bardist_` is deprecated and will be removed in a future version. "
-            "Please use `znorm_space_bardist_` instead.",
-            DeprecationWarning,
-            stacklevel=2,
-        )
-        return self.znorm_space_bardist_
-
-    @bardist_.setter
-    def bardist_(self, value: FullSupportBarDistribution) -> None:
-        """DEPRECATED: Setting `bardist_` is deprecated.
-        Use `znorm_space_bardist_`.
-        """
-        warnings.warn(
-            "`bardist_` is deprecated and will be removed in a future version. "
-            "Please use `znorm_space_bardist_` instead.",
-            DeprecationWarning,
-            stacklevel=2,
-        )
-        self.znorm_space_bardist_ = value
+    bardist_: FullSupportBarDistribution
 
 
 @dataclass(frozen=True, eq=True)
@@ -113,18 +83,11 @@ class PreprocessorConfig:
         categorical_name:
             Name of the categorical encoding method.
             Options: "none", "numeric", "onehot", "ordinal", "ordinal_shuffled", "none".
-        append_to_original: If set to "auto", this is dynamically set to
+        append_to_original: If True, the transformed features are appended
+            to the original features. If set to "auto", this is dynamically set to
             True if the number of features is less than 500, and False otherwise.
-            Note that if set to "auto" and `max_features_per_estimator` is set as well,
-            this flag will become False if the number of features is larger than
-            `max_features_per_estimator / 2`. If True, the transformed features are
-            appended to the original features, however both are capped at the
-            max_features_per_estimator threshold, this should be used with caution as a
-            given model might not be configured for it.
-        max_features_per_estimator: Maximum number of features per estimator. In case
-            the dataset has more features than this, the features are subsampled for
-            each estimator independently. If append to original is set to True we can
-            still have more features.
+            Defaults to False.
+        subsample_features: Fraction of features to subsample. -1 means no subsampling.
         global_transformer_name: Name of the global transformer to use.
     """
 
@@ -140,8 +103,6 @@ class PreprocessorConfig:
         "quantile_norm",
         "quantile_uni_fine",
         "quantile_norm_fine",
-        "squashing_scaler_default",
-        "squashing_scaler_max10",
         "robust",  # a standard sklearn robust scaler
         "kdi",
         "none",  # no transformation (only standardization in transformer)
@@ -187,15 +148,8 @@ class PreprocessorConfig:
         "ordinal_very_common_categories_shuffled",
     ] = "none"
     append_original: bool | Literal["auto"] = False
-    max_features_per_estimator: int = 500
-    global_transformer_name: (
-        Literal[
-            "scaler",
-            "svd",
-            "svd_quarter_components",
-        ]
-        | None
-    ) = None
+    subsample_features: float = -1
+    global_transformer_name: str | None = None
     differentiable: bool = False
 
     @override
@@ -203,7 +157,11 @@ class PreprocessorConfig:
         return (
             f"{self.name}_cat:{self.categorical_name}"
             + ("_and_none" if self.append_original else "")
-            + (f"_max_feats_per_est_{self.max_features_per_estimator}")
+            + (
+                f"_subsample_feats_{self.subsample_features}"
+                if self.subsample_features > 0
+                else ""
+            )
             + (
                 f"_global_transformer_{self.global_transformer_name}"
                 if self.global_transformer_name is not None
@@ -211,88 +169,65 @@ class PreprocessorConfig:
             )
         )
 
+    def to_dict(self) -> dict:
+        """Convert the config to a dictionary.
 
-def default_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
-    """Get default preprocessor configurations for classification.
-
-    These are the defaults used when training new models, which will then be stored in
-    the model checkpoint.
-
-    See `v2_classifier_preprocessor_configs()`, `v2_5_classifier_preprocessor_configs()`
-    for the preprocessing used earlier versions of the model.
-    """
-    return [
-        PreprocessorConfig(
-            name="squashing_scaler_default",
-            append_original=False,
-            categorical_name="ordinal_very_common_categories_shuffled",
-            global_transformer_name="svd_quarter_components",
-            max_features_per_estimator=500,
-        ),
-        PreprocessorConfig(
-            name="none",
-            categorical_name="numeric",
-            max_features_per_estimator=500,
-        ),
-    ]
-
-
-def default_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
-    """Default preprocessor configurations for regression.
-
-    These are the defaults used when training new models, which will then be stored in
-    the model checkpoint.
+        Returns:
+            Dictionary representation of the config.
+        """
+        return {
+            "name": self.name,
+            "categorical_name": self.categorical_name,
+            "append_original": self.append_original,
+            "subsample_features": self.subsample_features,
+            "global_transformer_name": self.global_transformer_name,
+            "differentiable": self.differentiable,
+        }
 
-    See `v2_regressor_preprocessor_configs()`, `v2_5_regressor_preprocessor_configs()`
-    for the preprocessing used earlier versions of the model.
-    """
-    return [
-        PreprocessorConfig(
-            name="quantile_uni_coarse",
-            append_original="auto",
-            categorical_name="numeric",
-            global_transformer_name=None,
-            max_features_per_estimator=500,
-        ),
-        PreprocessorConfig(
-            name="squashing_scaler_default",
-            append_original=False,
-            categorical_name="ordinal_very_common_categories_shuffled",
-            global_transformer_name="svd_quarter_components",
-            max_features_per_estimator=500,
-        ),
-    ]
+    @classmethod
+    def from_dict(cls, config_dict: dict) -> PreprocessorConfig:
+        """Create a config from a dictionary.
 
+        Args:
+            config_dict: Dictionary containing the config parameters.
 
-# Feature subsampling was disabled in v2, so choose a threshold that will never be
-# reached.
-_V2_FEATURE_SUBSAMPLING_THRESHOLD = 1_000_000
+        Returns:
+            PreprocessorConfig instance.
+        """
+        return cls(
+            name=config_dict["name"],
+            categorical_name=config_dict["categorical_name"],
+            append_original=config_dict["append_original"],
+            subsample_features=config_dict["subsample_features"],
+            global_transformer_name=config_dict["global_transformer_name"],
+            differentiable=config_dict.get("differentiable", False),
+        )
 
 
-def v2_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
-    """Get the preprocessor configuration for classification in v2 of the model."""
+def default_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
+    """Default preprocessor configurations for classification."""
     return [
         PreprocessorConfig(
             "quantile_uni_coarse",
             append_original="auto",
             categorical_name="ordinal_very_common_categories_shuffled",
             global_transformer_name="svd",
-            max_features_per_estimator=_V2_FEATURE_SUBSAMPLING_THRESHOLD,
+            subsample_features=-1,
         ),
         PreprocessorConfig(
             "none",
             categorical_name="numeric",
-            max_features_per_estimator=_V2_FEATURE_SUBSAMPLING_THRESHOLD,
+            subsample_features=-1,
         ),
     ]
 
 
-def v2_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
-    """Get the preprocessor configuration for regression in v2 of the model."""
+def default_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
+    """Default preprocessor configurations for regression."""
     return [
         PreprocessorConfig(
             "quantile_uni",
-            append_original=True,
+            append_original="auto",
             categorical_name="ordinal_very_common_categories_shuffled",
             global_transformer_name="svd",
         ),
@@ -300,44 +235,6 @@ def v2_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
     ]
 
 
-def v2_5_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
-    """Get the preprocessor configuration for classification in v2.5 of the model."""
-    return [
-        PreprocessorConfig(
-            name="squashing_scaler_default",
-            append_original=False,
-            categorical_name="ordinal_very_common_categories_shuffled",
-            global_transformer_name="svd_quarter_components",
-            max_features_per_estimator=500,
-        ),
-        PreprocessorConfig(
-            name="none",
-            categorical_name="numeric",
-            max_features_per_estimator=500,
-        ),
-    ]
-
-
-def v2_5_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
-    """Get the preprocessor configuration for regression in v2.5 of the model."""
-    return [
-        PreprocessorConfig(
-            name="quantile_uni_coarse",
-            append_original="auto",
-            categorical_name="numeric",
-            global_transformer_name=None,
-            max_features_per_estimator=500,
-        ),
-        PreprocessorConfig(
-            name="squashing_scaler_default",
-            append_original=False,
-            categorical_name="ordinal_very_common_categories_shuffled",
-            global_transformer_name="svd_quarter_components",
-            max_features_per_estimator=500,
-        ),
-    ]
-
-
 def generate_index_permutations(
     n: int,
     *,
@@ -382,11 +279,9 @@ class EnsembleConfig:
     """Configuration for an ensemble member.
 
     Attributes:
-        preprocess_config: Preprocessor configuration to use.
-        add_fingerprint_feature: Whether to add fingerprint features.
-        polynomial_features: Maximum number of polynomial features to add, if any.
         feature_shift_count: How much to shift the features columns.
-        feature_shift_decoder: How to shift features.
+        class_permutation: Permutation to apply to classes
+        preprocess_config: Preprocessor configuration to use.
         subsample_ix: Indices of samples to use for this ensemble member.
             If `None`, no subsampling is done.
     """
@@ -397,14 +292,12 @@ class EnsembleConfig:
     feature_shift_count: int
     feature_shift_decoder: Literal["shuffle", "rotate"] | None
     subsample_ix: npt.NDArray[np.int64] | None  # OPTIM: Could use uintp
-    # Internal index specifying which model to use for this ensemble member.
-    _model_index: int
 
     @classmethod
-    def generate_for_classification(  # noqa: PLR0913
+    def generate_for_classification(
         cls,
         *,
-        num_estimators: int,
+        n: int,
         subsample_size: int | float | None,
         max_index: int,
         add_fingerprint_feature: bool,
@@ -414,12 +307,11 @@ class EnsembleConfig:
         class_shift_method: Literal["rotate", "shuffle"] | None,
         n_classes: int,
         random_state: int | np.random.Generator | None,
-        num_models: int,
     ) -> list[ClassifierEnsembleConfig]:
         """Generate ensemble configurations for classification.
 
         Args:
-            num_estimators: Number of ensemble configurations to generate.
+            n: Number of ensemble configurations to generate.
             subsample_size:
                 Number of samples to subsample. If int, subsample that many
                 samples. If float, subsample that fraction of samples. If `None`, no
@@ -432,93 +324,80 @@ class EnsembleConfig:
             class_shift_method: How to shift classes for classpermutation.
             n_classes: Number of classes.
             random_state: Random number generator.
-            num_models: Number of models to use.
 
         Returns:
             List of ensemble configurations.
         """
         static_seed, rng = infer_random_state(random_state)
         start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
-        featshifts = np.arange(start, start + num_estimators)
-        featshifts = rng.choice(featshifts, size=num_estimators, replace=False)  # type: ignore
+        featshifts = np.arange(start, start + n)
+        featshifts = rng.choice(featshifts, size=n, replace=False)  # type: ignore
 
         if class_shift_method == "rotate":
             arange = np.arange(0, n_classes)
             shifts = rng.permutation(n_classes).tolist()
             class_permutations = [np.roll(arange, s) for s in shifts]
             class_permutations = [  # type: ignore
-                class_permutations[c] for c in rng.choice(n_classes, num_estimators)
+                class_permutations[c] for c in rng.choice(n_classes, n)
             ]
         elif class_shift_method == "shuffle":
-            noise = rng.random(
-                (num_estimators * CLASS_SHUFFLE_OVERESTIMATE_FACTOR, n_classes)
-            )
+            noise = rng.random((n * CLASS_SHUFFLE_OVERESTIMATE_FACTOR, n_classes))
             shufflings = np.argsort(noise, axis=1)
             uniqs = np.unique(shufflings, axis=0)
-            balance_count = num_estimators // len(uniqs)
+            balance_count = n // len(uniqs)
             class_permutations = balance(uniqs, balance_count)
-            rand_count = num_estimators % len(uniqs)
+            rand_count = n % len(uniqs)
             if rand_count > 0:
                 class_permutations += [  # type: ignore
                     uniqs[i] for i in rng.choice(len(uniqs), size=rand_count)
                 ]
         elif class_shift_method is None:
-            class_permutations = [None] * num_estimators  # type: ignore
+            class_permutations = [None] * n  # type: ignore
         else:
             raise ValueError(f"Unknown {class_shift_method=}")
 
         subsamples: list[None] | list[np.ndarray]
         if isinstance(subsample_size, (int, float)):
             subsamples = generate_index_permutations(
-                n=num_estimators,
+                n=n,
                 max_index=max_index,
                 subsample=subsample_size,
                 random_state=static_seed,
             )
         elif subsample_size is None:
-            subsamples = [None] * num_estimators  # type: ignore
+            subsamples = [None] * n  # type: ignore
         else:
             raise ValueError(
                 f"Invalid subsample_samples: {subsample_size}",
             )
 
-        balance_count = num_estimators // len(preprocessor_configs)
+        balance_count = n // len(preprocessor_configs)
 
         # Replicate each config balance_count times
         configs_ = balance(preprocessor_configs, balance_count)
+
         # Number still needed to reach n
-        leftover = num_estimators - len(configs_)
+        leftover = n - len(configs_)
+
         if leftover > 0:
             # the preprocessor configs should be ordered by performance
             configs_.extend(preprocessor_configs[:leftover])
 
-        # Models are simply cycled through for the estimators.
-        # This ensures that different preprocessings are applied to different models.
-        model_indices = [i % num_models for i in range(num_estimators)]
-
         return [
             ClassifierEnsembleConfig(
-                preprocess_config=preprocesses_config,
+                preprocess_config=preprocess_config,
                 feature_shift_count=featshift,
                 add_fingerprint_feature=add_fingerprint_feature,
                 polynomial_features=polynomial_features,
                 feature_shift_decoder=feature_shift_decoder,
                 subsample_ix=subsample_ix,
                 class_permutation=class_perm,
-                _model_index=model_index,
             )
-            for (
-                featshift,
-                preprocesses_config,
-                subsample_ix,
-                class_perm,
-                model_index,
-            ) in zip(
+            for featshift, preprocess_config, subsample_ix, class_perm in zip(
                 featshifts,
                 configs_,
                 subsamples,
                 class_permutations,
-                model_indices,
             )
         ]
 
@@ -526,7 +405,7 @@ class EnsembleConfig:
     def generate_for_regression(
         cls,
         *,
-        num_estimators: int,
+        n: int,
         subsample_size: int | float | None,
         max_index: int,
         add_fingerprint_feature: bool,
@@ -535,12 +414,11 @@ class EnsembleConfig:
         preprocessor_configs: Sequence[PreprocessorConfig],
         target_transforms: Sequence[TransformerMixin | Pipeline | None],
         random_state: int | np.random.Generator | None,
-        num_models: int,
     ) -> list[RegressorEnsembleConfig]:
         """Generate ensemble configurations for regression.
 
         Args:
-            num_estimators: Number of ensemble configurations to generate.
+            n: Number of ensemble configurations to generate.
             subsample_size:
                 Number of samples to subsample. If int, subsample that many
                 samples. If float, subsample that fraction of samples. If `None`, no
@@ -552,26 +430,25 @@ class EnsembleConfig:
             preprocessor_configs: Preprocessor configurations to use on the data.
             target_transforms: Target transformations to apply.
             random_state: Random number generator.
-            num_models: Number of models to use.
 
         Returns:
             List of ensemble configurations.
         """
         static_seed, rng = infer_random_state(random_state)
         start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
-        featshifts = np.arange(start, start + num_estimators)
-        featshifts = rng.choice(featshifts, size=num_estimators, replace=False)  # type: ignore
+        featshifts = np.arange(start, start + n)
+        featshifts = rng.choice(featshifts, size=n, replace=False)  # type: ignore
 
         subsamples: list[None] | list[np.ndarray]
         if isinstance(subsample_size, (int, float)):
             subsamples = generate_index_permutations(
-                n=num_estimators,
+                n=n,
                 max_index=max_index,
                 subsample=subsample_size,
                 random_state=static_seed,
             )
         elif subsample_size is None:
-            subsamples = [None] * num_estimators
+            subsamples = [None] * n
         else:
             raise ValueError(
                 f"Invalid subsample_samples: {subsample_size}",
@@ -579,20 +456,17 @@ class EnsembleConfig:
 
         # Get equal representation of all preprocessor configs
         combos = list(product(preprocessor_configs, target_transforms))
-        balance_count = num_estimators // len(combos)
+        balance_count = n // len(combos)
         configs_ = balance(combos, balance_count)
+
         # Number still needed to reach n
-        leftover = num_estimators - len(configs_)
+        leftover = n - len(configs_)
+
         if leftover > 0:
             # the preprocessor configs should be ordered by performance
             # TODO: what about the target transforms?
             configs_ += combos[:leftover]
 
-        # Models are simply cycled through for the estimators.
-        # This ensures that different preprocessings and target transformations are
-        # applied to different models.
-        model_indices = [i % num_models for i in range(num_estimators)]
-
         return [
             RegressorEnsembleConfig(
                 preprocess_config=preprocess_config,
@@ -602,16 +476,11 @@ class EnsembleConfig:
                 feature_shift_decoder=feature_shift_decoder,
                 subsample_ix=subsample_ix,
                 target_transform=target_transform,
-                _model_index=model_index,
             )
-            for featshift, subsample_ix, (
-                preprocess_config,
-                target_transform,
-            ), model_index in zip(
+            for featshift, subsample_ix, (preprocess_config, target_transform) in zip(
                 featshifts,
                 subsamples,
                 configs_,
-                model_indices,
             )
         ]
 
@@ -656,7 +525,7 @@ class EnsembleConfig:
                     ReshapeFeatureDistributionsStep(
                         transform_name=self.preprocess_config.name,
                         append_to_original=self.preprocess_config.append_original,
-                        max_features_per_estimator=self.preprocess_config.max_features_per_estimator,
+                        subsample_features=self.preprocess_config.subsample_features,
                         global_transformer_name=self.preprocess_config.global_transformer_name,
                         apply_to_categorical=(
                             self.preprocess_config.categorical_name == "numeric"
@@ -687,9 +556,6 @@ class EnsembleConfig:
 class ClassifierEnsembleConfig(EnsembleConfig):
     """Configuration for a classifier ensemble member.
 
-    Attributes:
-        class_permutation: Permutation to apply to classes
-
     See [EnsembleConfig][tabpfn.preprocessing.EnsembleConfig] for more details.
     """
 
@@ -756,9 +622,7 @@ def fit_preprocessing_one(
     return (config, preprocessor, res.X, y_train_processed, res.categorical_features)
 
 
-def transform_labels_one(
-    config: EnsembleConfig, y_train: np.ndarray | torch.Tensor
-) -> np.ndarray:
+def transform_labels_one(config, y_train):
     """Transform the labels for one ensemble config.
         for both regression or classification.
 
@@ -791,7 +655,7 @@ def fit_preprocessing(
     *,
     random_state: int | np.random.Generator | None,
     cat_ix: list[int],
-    n_preprocessing_jobs: int,
+    n_workers: int,  # noqa: ARG001
     parallel_mode: Literal["block", "as-ready", "in-order"],
 ) -> Iterator[
     tuple[
@@ -810,16 +674,7 @@ def fit_preprocessing(
         y_train: Training target.
         random_state: Random number generator.
         cat_ix: Indices of categorical features.
-        n_preprocessing_jobs: Number of worker processes to use.
-            If `1`, then the preprocessing is performed in the current process. This
-                avoids multiprocessing overheads, but may not be able to full saturate
-                the CPU. Note that the preprocessing itself will parallelise over
-                multiple cores, so one job is often enough.
-            If `>1`, then different estimators are dispatched to different proceses,
-                which allows more parallelism but incurs some overhead.
-            If `-1`, then creates as many workers as CPU cores. As each worker itself
-                uses multiple cores, this is likely too many.
-            It is best to select this value by benchmarking.
+        n_workers: Number of workers to use.
         parallel_mode:
             Parallel mode to use.
 
@@ -835,16 +690,32 @@ def fit_preprocessing(
     """
     _, rng = infer_random_state(random_state)
 
-    # Below we set batch_size to auto, but this could be further tuned.
+    # TODO: It seems like we really don't benefit from much more than 1,2,4 workers,
+    # even for the largest datasets from AutoMLBenchmark. Even then, the benefit is
+    # marginal. For now, we stick with single worker.
+    #
+    # The parameters worth tuning are `batch_size` and `n_jobs`
+    # * `n_jobs` - how many workers to spawn.
+    # * `batch_size` - how many tasks to send to a worker at once.
+    #
+    # For small datasets (for which this model is built for), it's quite hard to tune
+    # for increased performance and staying at 1 worker seems ideal. However for larger
+    # datasets, at the limit of what we support, having `len(configs) // 2` workers
+    # seemed good, with a `batch_size` of 2.
+    # NOTE: By setting `n_jobs` = 1, it effectively doesn't spawn anything and runs
+    # in-process
     if SUPPORTS_RETURN_AS:
         return_as = PARALLEL_MODE_TO_RETURN_AS[parallel_mode]
         executor = joblib.Parallel(
-            n_jobs=n_preprocessing_jobs,
+            n_jobs=1,
             return_as=return_as,
-            batch_size="auto",
+            batch_size="auto",  # type: ignore
         )
     else:
-        executor = joblib.Parallel(n_jobs=n_preprocessing_jobs, batch_size="auto")
+        executor = joblib.Parallel(
+            n_jobs=1,
+            batch_size="auto",  # type: ignore
+        )
     func = partial(fit_preprocessing_one, cat_ix=cat_ix)
     worker_func = joblib.delayed(func)
 
@@ -890,44 +761,38 @@ class DatasetCollectionWithPreprocessing(Dataset):
             `sklearn.model_selection.train_test_split`). It's used to split
             the raw data (X, y) into train and test sets. It will receive
             `X`, `y`, and `random_state` as arguments.
-        rng: A NumPy random number generator instance
+        rng (np.random.Generator): A NumPy random number generator instance
             used for generating the split seed and potentially within the
             preprocessing steps defined in the configs.
-        dataset_config_collection: A sequence containing dataset configuration objects.
-            Each object must hold the raw data (`X_raw`, `y_raw`), categorical feature
+        dataset_config_collection (
+            Sequence[Union[RegressorDatasetConfig, ClassifierDatasetConfig]]
+        ): A sequence containing dataset configuration objects. Each object
+            must hold the raw data (`X_raw`, `y_raw`), categorical feature
             indices (`cat_ix`), and the specific preprocessing configurations
             (`config`) for that dataset. Regression configs require additional
-            fields (`znorm_space_bardist_`).
-        n_preprocessing_jobs: The number of workers to use for potentially parallelized
-            preprocessing steps (passed to `fit_preprocessing`).
+            fields (`y_full_standardised`, `normalized_bardist_`).
+        n_workers (int, optional): The number of workers to use for potentially
+            parallelized preprocessing steps (passed to `fit_preprocessing`).
+            Defaults to 1.
 
     Attributes:
         configs (Sequence[Union[RegressorDatasetConfig, ClassifierDatasetConfig]]):
             Stores the input dataset configuration collection.
         split_fn (Callable): Stores the splitting function.
         rng (np.random.Generator): Stores the random number generator.
-        n_preprocessing_jobs (int): The number of worker processes that will be used for
-            the preprocessing.
+        n_workers (int): Stores the number of workers for preprocessing.
     """
 
-    def __init__(
-        self,
-        split_fn: Callable,
-        rng: np.random.Generator,
-        dataset_config_collection: Sequence[
-            RegressorDatasetConfig | ClassifierDatasetConfig
-        ],
-        n_preprocessing_jobs: int = 1,
-    ) -> None:
+    def __init__(self, split_fn, rng, dataset_config_collection, n_workers=1):
         self.configs = dataset_config_collection
         self.split_fn = split_fn
         self.rng = rng
-        self.n_preprocessing_jobs = n_preprocessing_jobs
+        self.n_workers = n_workers
 
     def __len__(self):
         return len(self.configs)
 
-    def __getitem__(self, index: int):  # noqa: C901, PLR0912
+    def __getitem__(self, index):  # noqa: C901, PLR0912
         """Retrieves, splits, and preprocesses the dataset config at the index.
 
         Performs train/test splitting and applies potentially multiple
@@ -974,12 +839,10 @@ class DatasetCollectionWithPreprocessing(Dataset):
                 * `cat_ixs` (List[Optional[List[int]]]): List of categorical feature
                   indices corresponding to each preprocessed X_train/X_test.
                 * `conf` (List): The list of preprocessing configurations used.
-                * `raw_space_bardist_` (FullSupportBarDistribution): Binning class
-                  for target variable (specific to the regression config). The
-                  calculations will be on raw data in raw space.
-                * `znorm_space_bardist_` (FullSupportBarDistribution): Binning class for
-                  target variable (specific to the regression config). The calculations
-                  will be on standardized data in znorm space.
+                * `normalized_bardist_` (FullSupportBarDistribution): Binning class
+                  for target variable (specific to the regression config).
+                * `bardist_` (FullSupportBarDistribution): Binning class for
+                  target variable (specific to the regression config).
                 * `x_test_raw` (torch.Tensor): Original, unprocessed test feature
                   tensor.
                 * `y_test_raw` (torch.Tensor): Original, unprocessed test target
@@ -1004,7 +867,7 @@ class DatasetCollectionWithPreprocessing(Dataset):
             x_full_raw = config.X_raw
             y_full_raw = config.y_raw
             cat_ix = config.cat_ix
-            znorm_space_bardist_ = config.znorm_space_bardist_
+            bardist_ = config.bardist_
         elif isinstance(config, ClassifierDatasetConfig):
             conf = config.config
             x_full_raw = config.X_raw
@@ -1021,7 +884,7 @@ class DatasetCollectionWithPreprocessing(Dataset):
 
         # Compute target variable Z-transform standardization
         # based on statistics of training set
-        # Note: Since we compute raw_space_bardist_ here,
+        # Note: Since we compute normalized_bardist_ here,
         # it is not set as an attribute of the Regressor class
         # This however makes also sense when considering that
         # this attribute changes on every dataset
@@ -1030,9 +893,8 @@ class DatasetCollectionWithPreprocessing(Dataset):
             train_std = np.std(y_train_raw)
             y_test_standardized = (y_test_raw - train_mean) / train_std
             y_train_standardized = (y_train_raw - train_mean) / train_std
-            raw_space_bardist_ = FullSupportBarDistribution(
-                znorm_space_bardist_.borders * train_std
-                + train_mean  # Inverse normalization back to raw space
+            normalized_bardist_ = FullSupportBarDistribution(
+                bardist_.borders * train_std + train_mean
             ).float()
 
         y_train = y_train_standardized if regression_task else y_train_raw
@@ -1043,7 +905,7 @@ class DatasetCollectionWithPreprocessing(Dataset):
             y_train=y_train,
             random_state=self.rng,
             cat_ix=cat_ix,
-            n_preprocessing_jobs=self.n_preprocessing_jobs,
+            n_workers=self.n_workers,
             parallel_mode="block",
         )
         (
@@ -1091,7 +953,7 @@ class DatasetCollectionWithPreprocessing(Dataset):
         # Also return raw_target variable because of flexiblity
         # in optimisation space -> see examples/
         # Also return corresponding target variable binning
-        # classes raw_space_bardist_ and znorm_space_bardist_
+        # classes normalized_bardist_ and bardist_
         if regression_task:
             return (
                 X_trains_preprocessed,
@@ -1100,8 +962,8 @@ class DatasetCollectionWithPreprocessing(Dataset):
                 y_test_standardized,
                 cat_ixs,
                 conf,
-                raw_space_bardist_,
-                znorm_space_bardist_,
+                normalized_bardist_,
+                bardist_,
                 x_test_raw,
                 y_test_raw,
             )
diff --git a/src/tabpfn/preprocessors/__init__.py b/src/tabpfn/preprocessors/__init__.py
deleted file mode 100644
index f92f77f..0000000
--- a/src/tabpfn/preprocessors/__init__.py
+++ /dev/null
@@ -1,49 +0,0 @@
-from tabpfn.preprocessors.adaptive_quantile_transformer import (
-    AdaptiveQuantileTransformer,
-)
-from tabpfn.preprocessors.add_fingerprint_features_step import (
-    AddFingerprintFeaturesStep,
-)
-from tabpfn.preprocessors.differentiable_z_norm_step import DifferentiableZNormStep
-from tabpfn.preprocessors.encode_categorical_features_step import (
-    EncodeCategoricalFeaturesStep,
-)
-from tabpfn.preprocessors.kdi_transformer import (
-    KDITransformerWithNaN,
-    get_all_kdi_transformers,
-)
-from tabpfn.preprocessors.nan_handling_polynomial_features_step import (
-    NanHandlingPolynomialFeaturesStep,
-)
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-    SequentialFeatureTransformer,
-)
-from tabpfn.preprocessors.remove_constant_features_step import (
-    RemoveConstantFeaturesStep,
-)
-from tabpfn.preprocessors.reshape_feature_distribution_step import (
-    ReshapeFeatureDistributionsStep,
-    get_all_reshape_feature_distribution_preprocessors,
-)
-from tabpfn.preprocessors.safe_power_transformer import SafePowerTransformer
-from tabpfn.preprocessors.shuffle_features_step import ShuffleFeaturesStep
-from tabpfn.preprocessors.squashing_scaler_transformer import SquashingScaler
-
-__all__ = [
-    "AdaptiveQuantileTransformer",
-    "AddFingerprintFeaturesStep",
-    "DifferentiableZNormStep",
-    "EncodeCategoricalFeaturesStep",
-    "FeaturePreprocessingTransformerStep",
-    "KDITransformerWithNaN",
-    "NanHandlingPolynomialFeaturesStep",
-    "RemoveConstantFeaturesStep",
-    "ReshapeFeatureDistributionsStep",
-    "SafePowerTransformer",
-    "SequentialFeatureTransformer",
-    "ShuffleFeaturesStep",
-    "SquashingScaler",
-    "get_all_kdi_transformers",
-    "get_all_reshape_feature_distribution_preprocessors",
-]
diff --git a/src/tabpfn/preprocessors/adaptive_quantile_transformer.py b/src/tabpfn/preprocessors/adaptive_quantile_transformer.py
deleted file mode 100644
index 93eea14..0000000
--- a/src/tabpfn/preprocessors/adaptive_quantile_transformer.py
+++ /dev/null
@@ -1,80 +0,0 @@
-"""Adaptive Quantile Transformer."""
-
-from __future__ import annotations
-
-from typing import Any
-from typing_extensions import override
-
-import numpy as np
-from sklearn.preprocessing import QuantileTransformer
-
-
-class AdaptiveQuantileTransformer(QuantileTransformer):
-    """A QuantileTransformer that automatically adapts the 'n_quantiles' parameter
-    based on the number of samples provided during the 'fit' method.
-
-    This fixes an issue in older versions of scikit-learn where the 'n_quantiles'
-    parameter could not exceed the number of samples in the input data.
-
-    This code prevents errors that occur when the requested 'n_quantiles' is
-    greater than the number of available samples in the input data (X).
-    This situation can arises because we first initialize the transformer
-    based on total samples and then subsample.
-    """
-
-    def __init__(
-        self,
-        *,
-        n_quantiles: int = 1_000,
-        subsample: int = 100_000,  # default in sklearn is 10k
-        **kwargs: Any,
-    ) -> None:
-        # Store the user's desired n_quantiles to use as an upper bound
-        self._user_n_quantiles = n_quantiles
-        # Initialize parent with this, but it will be adapted in fit
-        super().__init__(n_quantiles=n_quantiles, subsample=subsample, **kwargs)
-
-    @override
-    def fit(
-        self,
-        X: np.ndarray,
-        y: np.ndarray | None = None,
-    ) -> AdaptiveQuantileTransformer:
-        n_samples = X.shape[0]
-
-        # Adapt n_quantiles for this fit: min of user's preference and available samples
-        # Ensure n_quantiles is at least 1.
-        # We allow the number of quantiles to be a maximum of 20% of the subsample size
-        # because we found that the `np.nanpercentile()` function inside sklearn's
-        # QuantileTransformer takes a long time to compute when the ratio
-        # of `quantiles / subsample` is too high (roughly higher than 0.25).
-        effective_n_quantiles = max(
-            1,
-            min(
-                self._user_n_quantiles,
-                n_samples,
-                int(self.subsample * 0.2),
-            ),
-        )
-
-        # Set self.n_quantiles to the effective value BEFORE calling super().fit()
-        # This ensures the parent class uses the adapted value for fitting
-        # and self.n_quantiles will reflect the value used for the fit.
-        self.n_quantiles = effective_n_quantiles
-
-        # Convert Generator to RandomState if needed for sklearn compatibility
-        if isinstance(self.random_state, np.random.Generator):
-            seed = int(self.random_state.integers(0, 2**32))
-            self.random_state = np.random.RandomState(seed)
-        elif hasattr(self.random_state, "bit_generator"):
-            raise ValueError(
-                f"Unsupported random state type: {type(self.random_state)}. "
-                "Please provide an integer seed or np.random.RandomState object."
-            )
-
-        return super().fit(X, y)
-
-
-__all__ = [
-    "AdaptiveQuantileTransformer",
-]
diff --git a/src/tabpfn/preprocessors/add_fingerprint_features_step.py b/src/tabpfn/preprocessors/add_fingerprint_features_step.py
deleted file mode 100644
index 70a92a6..0000000
--- a/src/tabpfn/preprocessors/add_fingerprint_features_step.py
+++ /dev/null
@@ -1,87 +0,0 @@
-"""Add Fingerprint Features Step."""
-
-from __future__ import annotations
-
-import hashlib
-from typing_extensions import override
-
-import numpy as np
-import torch
-
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-)
-from tabpfn.utils import infer_random_state
-
-_CONSTANT = 10**12
-
-
-def _float_hash_arr(arr: np.ndarray) -> float:
-    _hash = int(hashlib.sha256(arr.tobytes()).hexdigest(), 16)
-    return _hash % _CONSTANT / _CONSTANT
-
-
-class AddFingerprintFeaturesStep(FeaturePreprocessingTransformerStep):
-    """Adds a fingerprint feature to the features based on hash of each row.
-
-    If `is_test = True`, it keeps the first hash even if there are collisions.
-    If `is_test = False`, it handles hash collisions by counting up and rehashing
-    until a unique hash is found.
-    The idea is basically to add a random feature to help the model distinguish between
-    identical rows. We use hashing to make sure the result does not depend on the order
-    of the rows.
-    """
-
-    def __init__(self, random_state: int | np.random.Generator | None = None):
-        super().__init__()
-        self.random_state = random_state
-
-    @override
-    def _fit(
-        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
-    ) -> list[int]:
-        _, rng = infer_random_state(self.random_state)
-        self.rnd_salt_ = int(rng.integers(0, 2**16))
-        return [*categorical_features]
-
-    @override
-    def _transform(  # type: ignore
-        self,
-        X: np.ndarray | torch.Tensor,
-        *,
-        is_test: bool = False,
-    ) -> np.ndarray | torch.Tensor:
-        X_det = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
-
-        # no detach necessary for numpy
-        X_h = np.zeros(X.shape[0], dtype=X_det.dtype)
-        if is_test:
-            # Keep the first hash even if there are collisions
-            salted_X = X_det + self.rnd_salt_
-            for i, row in enumerate(salted_X):
-                h = _float_hash_arr(row + self.rnd_salt_)
-                X_h[i] = h
-        else:
-            # Handle hash collisions by counting up and rehashing
-            seen_hashes = set()
-            salted_X = X_det + self.rnd_salt_
-            for i, row in enumerate(salted_X):
-                h = _float_hash_arr(row)
-                add_to_hash = 0
-                while h in seen_hashes and not np.isnan(row).all():
-                    add_to_hash += 1
-                    h = _float_hash_arr(row + add_to_hash)
-                X_h[i] = h
-                seen_hashes.add(h)
-
-        if isinstance(X, torch.Tensor):
-            return torch.cat(
-                [X, torch.from_numpy(X_h).float().reshape(-1, 1).to(X.device)], dim=1
-            )
-        else:  # noqa: RET505
-            return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)
-
-
-__all__ = [
-    "AddFingerprintFeaturesStep",
-]
diff --git a/src/tabpfn/preprocessors/differentiable_z_norm_step.py b/src/tabpfn/preprocessors/differentiable_z_norm_step.py
deleted file mode 100644
index 644479c..0000000
--- a/src/tabpfn/preprocessors/differentiable_z_norm_step.py
+++ /dev/null
@@ -1,38 +0,0 @@
-"""Differentiable Z-Norm Step."""
-
-from __future__ import annotations
-
-from typing_extensions import override
-
-import torch
-
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-)
-
-
-class DifferentiableZNormStep(FeaturePreprocessingTransformerStep):
-    """Differentiable Z-Norm Step."""
-
-    def __init__(self):
-        super().__init__()
-
-        self.means = torch.tensor([])
-        self.stds = torch.tensor([])
-
-    @override
-    def _fit(self, X: torch.Tensor, categorical_features: list[int]) -> list[int]:  # type: ignore
-        self.means = X.mean(dim=0, keepdim=True)
-        self.stds = X.std(dim=0, keepdim=True)
-        return categorical_features
-
-    @override
-    def _transform(self, X: torch.Tensor, *, is_test: bool = False) -> torch.Tensor:  # type: ignore
-        assert X.shape[1] == self.means.shape[1]
-        assert X.shape[1] == self.stds.shape[1]
-        return (X - self.means) / self.stds
-
-
-__all__ = [
-    "DifferentiableZNormStep",
-]
diff --git a/src/tabpfn/preprocessors/encode_categorical_features_step.py b/src/tabpfn/preprocessors/encode_categorical_features_step.py
deleted file mode 100644
index 40b1c5a..0000000
--- a/src/tabpfn/preprocessors/encode_categorical_features_step.py
+++ /dev/null
@@ -1,233 +0,0 @@
-"""Encode Categorical Features Step."""
-
-from __future__ import annotations
-
-import warnings
-from typing_extensions import override
-
-import numpy as np
-from sklearn.compose import ColumnTransformer
-from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
-
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-    TransformResult,
-)
-from tabpfn.utils import infer_random_state
-
-
-def _get_least_common_category_count(x_column: np.ndarray) -> int:
-    if len(x_column) == 0:
-        return 0
-    counts = np.unique(x_column, return_counts=True)[1]
-    return int(counts.min())
-
-
-class EncodeCategoricalFeaturesStep(FeaturePreprocessingTransformerStep):
-    """Encode Categorical Features Step."""
-
-    def __init__(
-        self,
-        categorical_transform_name: str = "ordinal",
-        random_state: int | np.random.Generator | None = None,
-    ):
-        super().__init__()
-        self.categorical_transform_name = categorical_transform_name
-        self.random_state = random_state
-
-        self.categorical_transformer_ = None
-
-    def _get_transformer(
-        self,
-        X: np.ndarray,
-        categorical_features: list[int],
-    ) -> tuple[ColumnTransformer | None, list[int]]:
-        if self.categorical_transform_name.startswith("ordinal"):
-            name = self.categorical_transform_name[len("ordinal") :]
-            # Create a column transformer
-            if name.startswith("_common_categories"):
-                name = name[len("_common_categories") :]
-                categorical_features = [
-                    i
-                    for i, col in enumerate(X.T)
-                    if i in categorical_features
-                    and _get_least_common_category_count(col) >= 10
-                ]
-            elif name.startswith("_very_common_categories"):
-                name = name[len("_very_common_categories") :]
-                categorical_features = [
-                    i
-                    for i, col in enumerate(X.T)
-                    if i in categorical_features
-                    and _get_least_common_category_count(col) >= 10
-                    and len(np.unique(col)) < (len(X) // 10)  # type: ignore
-                ]
-
-            assert name in ("_shuffled", ""), (
-                "unknown categorical transform name, should be 'ordinal'"
-                f" or 'ordinal_shuffled' it was {self.categorical_transform_name}"
-            )
-
-            ct = ColumnTransformer(
-                [
-                    (
-                        "ordinal_encoder",
-                        OrdinalEncoder(
-                            handle_unknown="use_encoded_value",
-                            unknown_value=np.nan,
-                        ),  # 'sparse' has been deprecated
-                        categorical_features,
-                    ),
-                ],
-                # The column numbers to be transformed
-                remainder="passthrough",  # Leave the rest of the columns untouched
-            )
-            return ct, categorical_features
-
-        if self.categorical_transform_name == "onehot":
-            # Create a column transformer
-            ct = ColumnTransformer(
-                [
-                    (
-                        "one_hot_encoder",
-                        OneHotEncoder(
-                            drop="if_binary",
-                            sparse_output=False,
-                            handle_unknown="ignore",
-                        ),
-                        categorical_features,
-                    ),
-                ],
-                # The column numbers to be transformed
-                remainder="passthrough",  # Leave the rest of the columns untouched
-            )
-            return ct, categorical_features
-
-        if self.categorical_transform_name in ("numeric", "none"):
-            return None, categorical_features
-        raise ValueError(
-            f"Unknown categorical transform {self.categorical_transform_name}",
-        )
-
-    @override
-    def _fit(
-        self,
-        X: np.ndarray,
-        categorical_features: list[int],
-    ) -> list[int]:
-        ct, categorical_features = self._get_transformer(X, categorical_features)
-        if ct is None:
-            self.categorical_transformer_ = None
-            return categorical_features
-
-        _, rng = infer_random_state(self.random_state)
-
-        if self.categorical_transform_name.startswith("ordinal"):
-            ct.fit(X)
-            categorical_features = list(range(len(categorical_features)))
-
-            self.random_mappings_ = {}
-            if self.categorical_transform_name.endswith("_shuffled"):
-                for col_ix in categorical_features:
-                    col_cats = len(
-                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
-                    )
-                    perm = rng.permutation(col_cats)
-                    self.random_mappings_[col_ix] = perm
-
-        elif self.categorical_transform_name == "onehot":
-            Xt = ct.fit_transform(X)
-            if Xt.size >= 1_000_000:
-                ct = None
-            else:
-                categorical_features = list(range(Xt.shape[1]))[
-                    ct.output_indices_["one_hot_encoder"]
-                ]
-        else:
-            raise ValueError(
-                f"Unknown categorical transform {self.categorical_transform_name}",
-            )
-
-        self.categorical_transformer_ = ct
-        return categorical_features
-
-    def _fit_transform(
-        self,
-        X: np.ndarray,
-        categorical_features: list[int],
-    ) -> tuple[np.ndarray, list[int]]:
-        ct, categorical_features = self._get_transformer(X, categorical_features)
-        if ct is None:
-            self.categorical_transformer_ = None
-            return X, categorical_features
-
-        _, rng = infer_random_state(self.random_state)
-
-        if self.categorical_transform_name.startswith("ordinal"):
-            Xt = ct.fit_transform(X)
-            categorical_features = list(range(len(categorical_features)))
-
-            self.random_mappings_ = {}
-            if self.categorical_transform_name.endswith("_shuffled"):
-                for col_ix in categorical_features:
-                    col_cats = len(
-                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
-                    )
-                    perm = rng.permutation(col_cats)
-                    self.random_mappings_[col_ix] = perm
-
-                    Xcol: np.ndarray = Xt[:, col_ix]  # type: ignore
-                    not_nan_mask = ~np.isnan(Xcol)
-                    Xcol[not_nan_mask] = perm[Xcol[not_nan_mask].astype(int)].astype(
-                        Xcol.dtype,
-                    )
-
-        elif self.categorical_transform_name == "onehot":
-            Xt = ct.fit_transform(X)
-            if Xt.size >= 1_000_000:
-                ct = None
-                Xt = X
-            else:
-                categorical_features = list(range(Xt.shape[1]))[
-                    ct.output_indices_["one_hot_encoder"]
-                ]
-        else:
-            raise ValueError(
-                f"Unknown categorical transform {self.categorical_transform_name}",
-            )
-
-        self.categorical_transformer_ = ct
-        return Xt, categorical_features  # type: ignore
-
-    @override
-    def fit_transform(
-        self,
-        X: np.ndarray,
-        categorical_features: list[int],
-    ) -> TransformResult:
-        Xt, cat_ix = self._fit_transform(X, categorical_features)
-        self.categorical_features_after_transform_ = cat_ix
-        return TransformResult(Xt, cat_ix)
-
-    @override
-    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
-        if self.categorical_transformer_ is None:
-            return X
-
-        with warnings.catch_warnings():
-            warnings.filterwarnings(
-                "ignore", message=".*Found unknown categories in col.*"
-            )  # These warnings are expected when transforming test data
-            transformed = self.categorical_transformer_.transform(X)
-        if self.categorical_transform_name.endswith("_shuffled"):
-            for col, mapping in self.random_mappings_.items():
-                not_nan_mask = ~np.isnan(transformed[:, col])  # type: ignore
-                transformed[:, col][not_nan_mask] = mapping[
-                    transformed[:, col][not_nan_mask].astype(int)
-                ].astype(transformed[:, col].dtype)
-        return transformed  # type: ignore
-
-
-__all__ = [
-    "EncodeCategoricalFeaturesStep",
-]
diff --git a/src/tabpfn/preprocessors/kdi_transformer.py b/src/tabpfn/preprocessors/kdi_transformer.py
deleted file mode 100644
index f598f7e..0000000
--- a/src/tabpfn/preprocessors/kdi_transformer.py
+++ /dev/null
@@ -1,113 +0,0 @@
-"""KDI Transformer with NaN."""
-
-from __future__ import annotations
-
-from typing import Any
-
-import numpy as np
-import torch
-from sklearn.preprocessing import (
-    PowerTransformer,
-)
-
-try:
-    from kditransform import KDITransformer
-
-    # This import fails on some systems, due to problems with numba
-except ImportError:
-    KDITransformer = PowerTransformer  # fallback to avoid error
-
-
-ALPHAS = (
-    0.05,
-    0.1,
-    0.2,
-    0.25,
-    0.3,
-    0.4,
-    0.5,
-    0.6,
-    0.8,
-    1.0,
-    1.2,
-    1.5,
-    1.8,
-    2.0,
-    2.5,
-    3.0,
-    5.0,
-)
-
-
-class KDITransformerWithNaN(KDITransformer):
-    """KDI transformer that can handle NaN values. It performs KDI with NaNs replaced by
-    mean values and then fills the NaN values with NaNs after the transformation.
-    """
-
-    def _more_tags(self) -> dict:
-        return {"allow_nan": True}
-
-    def fit(
-        self,
-        X: torch.Tensor | np.ndarray,
-        y: Any | None = None,
-    ) -> KDITransformerWithNaN:
-        """Fit the transformer."""
-        if isinstance(X, torch.Tensor):
-            X = X.cpu().numpy()
-
-        # If all-nan or empty, nanmean returns nan.
-        self.imputation_values_ = np.nan_to_num(np.nanmean(X, axis=0), nan=0)
-        X = np.nan_to_num(X, nan=self.imputation_values_)
-
-        return super().fit(X, y)  # type: ignore
-
-    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
-        """Transform the data."""
-        # if tensor convert to numpy
-        if isinstance(X, torch.Tensor):
-            X = X.cpu().numpy()
-
-        # Calculate the NaN mask for the current dataset
-        nan_mask = np.isnan(X)
-
-        # Replace NaNs with the mean of columns
-        X = np.nan_to_num(X, nan=self.imputation_values_)
-
-        # Apply the transformation
-        X = super().transform(X)
-
-        # Reintroduce NaN values based on the current dataset's mask
-        X[nan_mask] = np.nan
-
-        return X  # type: ignore
-
-
-def get_all_kdi_transformers() -> dict[str, KDITransformerWithNaN]:
-    """Get all KDI transformers."""
-    try:
-        all_preprocessors = {
-            "kdi": KDITransformerWithNaN(alpha=1.0, output_distribution="normal"),
-            "kdi_uni": KDITransformerWithNaN(
-                alpha=1.0,
-                output_distribution="uniform",
-            ),
-        }
-        for alpha in ALPHAS:
-            all_preprocessors[f"kdi_alpha_{alpha}"] = KDITransformerWithNaN(
-                alpha=alpha,
-                output_distribution="normal",
-            )
-            all_preprocessors[f"kdi_alpha_{alpha}_uni"] = KDITransformerWithNaN(
-                alpha=alpha,
-                output_distribution="uniform",
-            )
-        return all_preprocessors
-    except Exception:  # noqa: BLE001
-        return {}
-
-
-__all__ = [
-    "KDITransformerWithNaN",
-    "get_all_kdi_transformers",
-]
diff --git a/src/tabpfn/preprocessors/nan_handling_polynomial_features_step.py b/src/tabpfn/preprocessors/nan_handling_polynomial_features_step.py
deleted file mode 100644
index 729e3af..0000000
--- a/src/tabpfn/preprocessors/nan_handling_polynomial_features_step.py
+++ /dev/null
@@ -1,95 +0,0 @@
-"""Nan Handling Polynomial Features Step."""
-
-from __future__ import annotations
-
-from typing_extensions import override
-
-import numpy as np
-from sklearn.preprocessing import (
-    StandardScaler,
-)
-
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-)
-from tabpfn.utils import infer_random_state
-
-
-class NanHandlingPolynomialFeaturesStep(FeaturePreprocessingTransformerStep):
-    """Nan Handling Polynomial Features Step."""
-
-    def __init__(
-        self,
-        *,
-        max_features: int | None = None,
-        random_state: int | np.random.Generator | None = None,
-    ):
-        super().__init__()
-
-        self.max_poly_features = max_features
-        self.random_state = random_state
-
-        self.poly_factor_1_idx: np.ndarray | None = None
-        self.poly_factor_2_idx: np.ndarray | None = None
-
-        self.standardizer = StandardScaler(with_mean=False)
-
-    @override
-    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
-        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"
-        _, rng = infer_random_state(self.random_state)
-
-        if X.shape[0] == 0 or X.shape[1] == 0:
-            return [*categorical_features]
-
-        # How many polynomials can we create?
-        n_polynomials = (X.shape[1] * (X.shape[1] - 1)) // 2 + X.shape[1]
-        n_polynomials = (
-            min(self.max_poly_features, n_polynomials)
-            if self.max_poly_features
-            else n_polynomials
-        )
-
-        X = self.standardizer.fit_transform(X)
-
-        # Randomly select the indices of the factors
-        self.poly_factor_1_idx = rng.choice(
-            np.arange(0, X.shape[1]),
-            size=n_polynomials,
-            replace=True,
-        )
-        self.poly_factor_2_idx = np.ones_like(self.poly_factor_1_idx) * -1
-        for i in range(len(self.poly_factor_1_idx)):
-            while self.poly_factor_2_idx[i] == -1:
-                poly_factor_1_ = self.poly_factor_1_idx[i]
-                # indices of the factors that have already been used
-                used_indices = self.poly_factor_2_idx[
-                    self.poly_factor_1_idx == poly_factor_1_
-                ]
-                # remaining indices, only factors with higher index can be selected
-                # to avoid duplicates
-                indices_ = set(range(poly_factor_1_, X.shape[1])) - set(
-                    used_indices.tolist(),
-                )
-                if len(indices_) == 0:
-                    self.poly_factor_1_idx[i] = rng.choice(
-                        np.arange(0, X.shape[1]),
-                        size=1,
-                    )
-                    continue
-                self.poly_factor_2_idx[i] = rng.choice(list(indices_), size=1)
-
-        return categorical_features
-
-    @override
-    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
-        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"
-
-        if X.shape[0] == 0 or X.shape[1] == 0:
-            return X
-
-        X = self.standardizer.transform(X)  # type: ignore
-
-        poly_features_xs = X[:, self.poly_factor_1_idx] * X[:, self.poly_factor_2_idx]
-
-        return np.hstack((X, poly_features_xs))
diff --git a/src/tabpfn/preprocessors/preprocessing_helpers.py b/src/tabpfn/preprocessors/preprocessing_helpers.py
deleted file mode 100644
index 14ad710..0000000
--- a/src/tabpfn/preprocessors/preprocessing_helpers.py
+++ /dev/null
@@ -1,310 +0,0 @@
-"""Feature Preprocessing Transformer Step."""
-
-from __future__ import annotations
-
-from abc import abstractmethod
-from collections import UserList
-from collections.abc import Callable, Iterable, Sequence
-from typing import TYPE_CHECKING, Any, NamedTuple
-from typing_extensions import Self, override
-
-if TYPE_CHECKING:
-    import torch
-
-    from tabpfn.classifier import XType, YType
-
-
-import numpy as np
-import pandas as pd
-from sklearn.base import (
-    BaseEstimator,
-    OneToOneFeatureMixin,
-    check_is_fitted,
-)
-from sklearn.compose import ColumnTransformer, make_column_selector
-from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
-
-from tabpfn.constants import DEFAULT_NUMPY_PREPROCESSING_DTYPE
-
-
-class TransformResult(NamedTuple):
-    """Result of a feature preprocessing step."""
-
-    X: np.ndarray | torch.Tensor
-    categorical_features: list[int]
-
-
-# TODO(eddiebergman): I'm sure there's a way to handle this when using dataframes.
-class FeaturePreprocessingTransformerStep:
-    """Base class for feature preprocessing steps.
-
-    It's main abstraction is really just to provide categorical indices along the
-    pipeline.
-    """
-
-    categorical_features_after_transform_: list[int]
-
-    def fit_transform(
-        self,
-        X: np.ndarray,
-        categorical_features: list[int],
-    ) -> TransformResult:
-        """Fits the preprocessor and transforms the data."""
-        self.fit(X, categorical_features)
-        # TODO(eddiebergman): If we could get rid of this... anywho, needed for
-        # the AddFingerPrint
-        result = self._transform(X, is_test=False)
-        return TransformResult(result, self.categorical_features_after_transform_)
-
-    @abstractmethod
-    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
-        """Underlying method of the preprocessor to implement by subclassses.
-
-        Args:
-            X: 2d array of shape (n_samples, n_features)
-            categorical_features: list of indices of categorical feature.
-
-        Returns:
-            list of indices of categorical features after the transform.
-        """
-        raise NotImplementedError
-
-    def fit(self, X: np.ndarray, categorical_features: list[int]) -> Self:
-        """Fits the preprocessor.
-
-        Args:
-            X: 2d array of shape (n_samples, n_features)
-            categorical_features: list of indices of categorical feature.
-        """
-        self.categorical_features_after_transform_ = self._fit(X, categorical_features)
-        assert self.categorical_features_after_transform_ is not None, (
-            "_fit should have returned a list of the indexes of the categorical"
-            "features after the transform."
-        )
-        return self
-
-    @abstractmethod
-    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
-        """Underlying method of the preprocessor to implement by subclassses.
-
-        Args:
-            X: 2d array of shape (n_samples, n_features)
-            is_test: Should be removed, used for the `AddFingerPrint` step.
-
-        Returns:
-            2d np.ndarray of shape (n_samples, new n_features)
-        """
-        raise NotImplementedError
-
-    def transform(self, X: np.ndarray) -> TransformResult:
-        """Transforms the data.
-
-        Args:
-            X: 2d array of shape (n_samples, n_features).
-        """
-        # TODO: Get rid of this, it's always test in `transform`
-        result = self._transform(X, is_test=True)
-        return TransformResult(result, self.categorical_features_after_transform_)
-
-
-class SequentialFeatureTransformer(UserList):
-    """A transformer that applies a sequence of feature preprocessing steps.
-    This is very related to sklearn's Pipeline, but it is designed to work with
-    categorical_features lists that are always passed on.
-
-    Currently this class is only used once, thus this could also be made
-    less general if needed.
-    """
-
-    def __init__(self, steps: list[FeaturePreprocessingTransformerStep]):
-        super().__init__(steps)
-        self.steps = steps
-        self.categorical_features_: list[int] | None = None
-
-    def fit_transform(
-        self,
-        X: np.ndarray | torch.Tensor,
-        categorical_features: list[int],
-    ) -> TransformResult:
-        """Fit and transform the data using the fitted pipeline.
-
-        Args:
-            X: 2d array of shape (n_samples, n_features)
-            categorical_features: list of indices of categorical features.
-        """
-        for step in self.steps:
-            X, categorical_features = step.fit_transform(X, categorical_features)
-            assert isinstance(categorical_features, list), (
-                f"The {step=} must return list of categorical features,"
-                f" but {type(step)} returned {categorical_features}"
-            )
-
-        self.categorical_features_ = categorical_features
-        return TransformResult(X, categorical_features)
-
-    def fit(
-        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
-    ) -> Self:
-        """Fit all the steps in the pipeline.
-
-        Args:
-            X: 2d array of shape (n_samples, n_features)
-            categorical_features: list of indices of categorical feature.
-        """
-        assert len(self) > 0, (
-            "The SequentialFeatureTransformer must have at least one step."
-        )
-        self.fit_transform(X, categorical_features)
-        return self
-
-    def transform(self, X: np.ndarray) -> TransformResult:
-        """Transform the data using the fitted pipeline.
-
-        Args:
-            X: 2d array of shape (n_samples, n_features).
-        """
-        assert len(self) > 0, (
-            "The SequentialFeatureTransformer must have at least one step."
-        )
-        assert self.categorical_features_ is not None, (
-            "The SequentialFeatureTransformer must be fit before it"
-            " can be used to transform."
-        )
-        categorical_features = []
-        for step in self:
-            X, categorical_features = step.transform(X)
-
-        assert categorical_features == self.categorical_features_, (
-            f"Expected categorical features {self.categorical_features_},"
-            f"but got {categorical_features}"
-        )
-        return TransformResult(X, categorical_features)
-
-
-class OrderPreservingColumnTransformer(ColumnTransformer):
-    """An ColumnTransformer that preserves the column order after transformation."""
-
-    def __init__(
-        self,
-        transformers: Sequence[
-            tuple[
-                str,
-                BaseEstimator,
-                str
-                | int
-                | slice
-                | Iterable[str | int]
-                | Callable[[Any], Iterable[str | int]],
-            ]
-        ],
-        **kwargs: Any,
-    ):
-        """Implementation base on https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html.
-
-        Parameters
-        ----------
-        transformers : sequence of (name, transformer, columns) tuples
-            List of (name, transformer, columns) tuples specifying the transformers.
-        **kwargs : additional keyword arguments
-            Passed to sklearn.compose.ColumnTransformer.
-        """
-        super().__init__(transformers=transformers, **kwargs)
-
-        # Check if there is a single transformer, of subtype OneToOneFeatureMixin
-        assert all(
-            isinstance(t, OneToOneFeatureMixin)
-            for name, t, _ in transformers
-            if name != "remainder"
-        ), (
-            "OrderPreservingColumnTransformer currently only supports transformers "
-            "that are instances of OneToOneFeatureMixin."
-        )
-
-        assert len([t for name, _, t in transformers if name != "remainder"]) <= 1, (
-            "OrderPreservingColumnTransformer only supports up to one transformer."
-        )
-
-    @override
-    def transform(self, X: XType, **kwargs: dict[str, Any]) -> XType:
-        original_columns = (
-            X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
-        )
-        X_t = super().transform(X, **kwargs)
-        return self._preserve_order(X=X_t, original_columns=original_columns)
-
-    @override
-    def fit_transform(
-        self, X: XType, y: YType = None, **kwargs: dict[str, Any]
-    ) -> XType:
-        original_columns = (
-            X.columns if isinstance(X, pd.DataFrame) else range(X.shape[-1])
-        )
-        X_t = super().fit_transform(X, y, **kwargs)
-        return self._preserve_order(X=X_t, original_columns=original_columns)
-
-    def _preserve_order(
-        self, X: XType, original_columns: list | range | pd.Index
-    ) -> XType:
-        check_is_fitted(self)
-        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D (shape={X.shape})"
-        for name, _, col_subset in reversed(self.transformers_):
-            if (
-                len(col_subset) > 0
-                and len(col_subset) < X.shape[-1]
-                and name != "remainder"
-            ):
-                col_subset_list = list(col_subset)
-                # Map original columns to indices in the transformed array
-                transformed_columns = col_subset_list + [
-                    c for c in original_columns if c not in col_subset_list
-                ]
-                indices = [transformed_columns.index(c) for c in original_columns]
-                # restore the column order from before the transfomer has been applied
-                X = X.iloc[:, indices] if isinstance(X, pd.DataFrame) else X[:, indices]
-        return X
-
-
-def get_ordinal_encoder(
-    *,
-    numpy_dtype: np.floating = DEFAULT_NUMPY_PREPROCESSING_DTYPE,  # type: ignore
-) -> ColumnTransformer:
-    """Create a ColumnTransformer that ordinally encodes string/category columns."""
-    oe = OrdinalEncoder(
-        # TODO: Could utilize the categorical dtype values directly instead of "auto"
-        categories="auto",
-        dtype=numpy_dtype,  # type: ignore
-        handle_unknown="use_encoded_value",
-        unknown_value=-1,
-        encoded_missing_value=np.nan,  # Missing stays missing
-    )
-
-    # Documentation of sklearn, deferring to pandas is misleading here. It's done
-    # using a regex on the type of the column, and using `object`, `"object"` and
-    # `np.object` will not pick up strings.
-    to_convert = ["category", "string"]
-
-    # When using a ColumnTransformer with inner transformers applied to only a subset of
-    # columns, the original column order of the data is not preserved. Because we do not
-    # update the categorical indices after encoding, these indices may no longer align
-    # with the true categorical columns.
-
-    # Subsequent components rely on these categorical indices. For instance:
-    # - QuantileTransformer should only be applied to numerical features.
-    # - EncodeCategoricalFeaturesStep should be applied to all categorical features.
-
-    # Despite the column shuffling introduced by the vanilla ColumnTransformer, we
-    # observed better overall performance when using it. Therefore, we keep it.
-
-    return ColumnTransformer(
-        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
-        remainder=FunctionTransformer(),
-        sparse_threshold=0.0,
-        verbose_feature_names_out=False,
-    )
-
-
-__all__ = [
-    "FeaturePreprocessingTransformerStep",
-    "SequentialFeatureTransformer",
-    "TransformResult",
-]
diff --git a/src/tabpfn/preprocessors/remove_constant_features_step.py b/src/tabpfn/preprocessors/remove_constant_features_step.py
deleted file mode 100644
index 48a48bd..0000000
--- a/src/tabpfn/preprocessors/remove_constant_features_step.py
+++ /dev/null
@@ -1,49 +0,0 @@
-"""Remove Constant Features Step."""
-
-from __future__ import annotations
-
-from typing_extensions import override
-
-import numpy as np
-import torch
-
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-)
-
-
-class RemoveConstantFeaturesStep(FeaturePreprocessingTransformerStep):
-    """Remove features that are constant in the training data."""
-
-    def __init__(self) -> None:
-        super().__init__()
-        self.sel_: list[bool] | None = None
-
-    @override
-    def _fit(  # type: ignore
-        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
-    ) -> list[int]:
-        if isinstance(X, torch.Tensor):
-            sel_ = torch.max(X[0:1, :] != X, dim=0)[0].cpu()
-        else:
-            sel_ = ((X[0:1, :] == X).mean(axis=0) < 1.0).tolist()
-
-        if not any(sel_):
-            raise ValueError(
-                "All features are constant and would have been removed!"
-                " Unable to predict using TabPFN.",
-            )
-        self.sel_ = sel_
-
-        return [
-            new_idx
-            for new_idx, idx in enumerate(np.where(sel_)[0])
-            if idx in categorical_features
-        ]
-
-    @override
-    def _transform(
-        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
-    ) -> np.ndarray:
-        assert self.sel_ is not None, "You must call fit first"
-        return X[:, self.sel_]
diff --git a/src/tabpfn/preprocessors/reshape_feature_distribution_step.py b/src/tabpfn/preprocessors/reshape_feature_distribution_step.py
deleted file mode 100644
index 02c1bc3..0000000
--- a/src/tabpfn/preprocessors/reshape_feature_distribution_step.py
+++ /dev/null
@@ -1,608 +0,0 @@
-"""Reshape the feature distributions using different transformations."""
-
-from __future__ import annotations
-
-import contextlib
-from copy import deepcopy
-from typing import TYPE_CHECKING, Literal, TypeVar
-from typing_extensions import override
-
-import numpy as np
-from scipy.stats import shapiro
-from sklearn.compose import ColumnTransformer, make_column_selector
-from sklearn.decomposition import TruncatedSVD
-from sklearn.impute import SimpleImputer
-from sklearn.pipeline import FeatureUnion, Pipeline
-from sklearn.preprocessing import (
-    FunctionTransformer,
-    MinMaxScaler,
-    PowerTransformer,
-    RobustScaler,
-    StandardScaler,
-)
-
-from tabpfn.preprocessors.adaptive_quantile_transformer import (
-    AdaptiveQuantileTransformer,
-)
-from tabpfn.preprocessors.kdi_transformer import (
-    KDITransformerWithNaN,
-    get_all_kdi_transformers,
-)
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-    TransformResult,
-)
-from tabpfn.preprocessors.safe_power_transformer import SafePowerTransformer
-from tabpfn.preprocessors.squashing_scaler_transformer import SquashingScaler
-from tabpfn.utils import infer_random_state
-
-if TYPE_CHECKING:
-    from sklearn.base import TransformerMixin
-
-T = TypeVar("T")
-
-
-def _identity(x: T) -> T:
-    return x
-
-
-def _inf_to_nan_func(x: np.ndarray) -> np.ndarray:
-    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)
-
-
-def _exp_minus_1(x: np.ndarray) -> np.ndarray:
-    return np.exp(x) - 1  # type: ignore
-
-
-inf_to_nan_transformer = FunctionTransformer(
-    func=_inf_to_nan_func,
-    inverse_func=_identity,
-    check_inverse=False,
-)
-nan_impute_transformer = SimpleImputer(
-    missing_values=np.nan,
-    strategy="mean",
-    # keep empty features for inverse to function
-    keep_empty_features=True,
-)
-nan_impute_transformer.inverse_transform = (
-    _identity  # do not inverse np.nan values.  # type: ignore
-)
-
-_make_finite_transformer = [
-    ("inf_to_nan", inf_to_nan_transformer),
-    ("nan_impute", nan_impute_transformer),
-]
-
-
-def _make_standard_scaler_safe(
-    _name_scaler_tuple: tuple[str, TransformerMixin],
-    *,
-    no_name: bool = False,
-) -> Pipeline:
-    # Make sure that all data that enters and leaves a scaler is finite.
-    # This is needed in edge cases where, for example, a division by zero
-    # occurs while scaling or when the input contains not number values.
-    return Pipeline(
-        steps=[
-            *[(n + "_pre ", deepcopy(t)) for n, t in _make_finite_transformer],
-            ("placeholder", _name_scaler_tuple) if no_name else _name_scaler_tuple,
-            *[(n + "_post", deepcopy(t)) for n, t in _make_finite_transformer],
-        ],
-    )
-
-
-def _make_box_cox_safe(input_transformer: TransformerMixin | Pipeline) -> Pipeline:
-    """Make box cox save.
-
-    The Box-Cox transformation can only be applied to strictly positive data.
-    With first MinMax scaling, we achieve this without loss of function.
-    Additionally, for test data, we also need clipping.
-    """
-    return Pipeline(
-        steps=[
-            ("mm", MinMaxScaler(feature_range=(0.1, 1), clip=True)),
-            ("box_cox", input_transformer),
-        ],
-    )
-
-
-def _add_safe_standard_to_safe_power_without_standard(
-    input_transformer: TransformerMixin,
-) -> Pipeline:
-    """In edge cases PowerTransformer can create inf values and similar. Then, the post
-    standard scale crashes. This fixes this issue.
-    """
-    return Pipeline(
-        steps=[
-            ("input_transformer", input_transformer),
-            ("standard", _make_standard_scaler_safe(("standard", StandardScaler()))),
-        ],
-    )
-
-
-def _skew(x: np.ndarray) -> float:
-    """skewness: 3 * (mean - median) / std."""
-    return float(3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0))
-
-
-class ReshapeFeatureDistributionsStep(FeaturePreprocessingTransformerStep):
-    """Reshape the feature distributions using different transformations."""
-
-    APPEND_TO_ORIGINAL_THRESHOLD = 500
-
-    @staticmethod
-    def get_column_types(X: np.ndarray) -> list[str]:
-        """Returns a list of column types for the given data, that indicate how
-        the data should be preprocessed.
-        """
-        # TODO(eddiebergman): Bad to keep calling skew again and again here...
-        column_types = []
-        for col in range(X.shape[1]):
-            if np.unique(X[:, col]).size < 10:
-                column_types.append(f"ordinal_{col}")
-            elif (
-                _skew(X[:, col]) > 1.1
-                and np.min(X[:, col]) >= 0
-                and np.max(X[:, col]) <= 1
-            ):
-                column_types.append(f"skewed_pos_1_0_{col}")
-            elif _skew(X[:, col]) > 1.1 and np.min(X[:, col]) > 0:
-                column_types.append(f"skewed_pos_{col}")
-            elif _skew(X[:, col]) > 1.1:
-                column_types.append(f"skewed_{col}")
-            elif shapiro(X[0:3000, col]).statistic > 0.95:
-                column_types.append(f"normal_{col}")
-            else:
-                column_types.append(f"other_{col}")
-        return column_types
-
-    def __init__(
-        self,
-        *,
-        transform_name: str = "safepower",
-        apply_to_categorical: bool = False,
-        append_to_original: bool | Literal["auto"] = False,
-        max_features_per_estimator: int = 500,
-        global_transformer_name: str | None = None,
-        random_state: int | np.random.Generator | None = None,
-    ):
-        super().__init__()
-
-        if max_features_per_estimator <= 0:
-            raise ValueError("max_features_per_estimator must be a positive integer.")
-
-        self.transform_name = transform_name
-        self.apply_to_categorical = apply_to_categorical
-        self.append_to_original = append_to_original
-        self.random_state = random_state
-        self.max_features_per_estimator = max_features_per_estimator
-        self.global_transformer_name = global_transformer_name
-        self.transformer_: Pipeline | ColumnTransformer | None = None
-
-    def _set_transformer_and_cat_ix(  # noqa: PLR0912
-        self,
-        n_samples: int,
-        n_features: int,
-        categorical_features: list[int],
-    ) -> tuple[Pipeline | ColumnTransformer, list[int]]:
-        if "adaptive" in self.transform_name:
-            raise NotImplementedError("Adaptive preprocessing raw removed.")
-
-        static_seed, rng = infer_random_state(self.random_state)
-
-        all_preprocessors = get_all_reshape_feature_distribution_preprocessors(
-            n_samples,
-            random_state=static_seed,
-        )
-        if n_features > self.max_features_per_estimator:
-            subsample_features = self.max_features_per_estimator
-            self.subsampled_features_ = rng.choice(
-                list(range(n_features)),
-                subsample_features,
-                replace=False,
-            )
-            categorical_features = [
-                new_idx
-                for new_idx, idx in enumerate(self.subsampled_features_)
-                if idx in categorical_features
-            ]
-            n_features = subsample_features
-        else:
-            self.subsampled_features_ = np.arange(n_features)
-
-        if (
-            self.global_transformer_name is not None
-            and self.global_transformer_name != "None"
-            and not (
-                self.global_transformer_name in ["svd", "svd_quarter_components"]
-                and n_features < 2
-            )
-        ):
-            global_transformer_ = get_all_global_transformers(
-                n_samples,
-                n_features,
-                random_state=static_seed,
-            )[self.global_transformer_name]
-        else:
-            global_transformer_ = None
-
-        all_feats_ix = list(range(n_features))
-        transformers = []
-
-        numerical_ix = [i for i in range(n_features) if i not in categorical_features]
-
-        append_decision = (
-            n_features < self.APPEND_TO_ORIGINAL_THRESHOLD
-            and n_features < (self.max_features_per_estimator / 2)
-        )
-        self.append_to_original = (
-            append_decision
-            if self.append_to_original == "auto"
-            else self.append_to_original
-        )
-
-        # -------- Append to original ------
-        # If we append to original, all the categorical indices are kept in place
-        # as the first transform is a passthrough on the whole X as it is above
-        if self.append_to_original and self.apply_to_categorical:
-            trans_ixs = categorical_features + numerical_ix
-            transformers.append(("original", "passthrough", all_feats_ix))
-            cat_ix = categorical_features  # Exist as they are in original
-
-        elif self.append_to_original and not self.apply_to_categorical:
-            trans_ixs = numerical_ix
-            # Includes the categoricals passed through
-            transformers.append(("original", "passthrough", all_feats_ix))
-            cat_ix = categorical_features  # Exist as they are in original
-
-        # -------- Don't append to original ------
-        # We only have categorical indices if we don't transform them
-        # The first transformer will be a passthrough on the categorical indices
-        # Making them the first
-        elif not self.append_to_original and self.apply_to_categorical:
-            trans_ixs = categorical_features + numerical_ix
-            cat_ix = []  # We have none left, they've been transformed
-
-        elif not self.append_to_original and not self.apply_to_categorical:
-            trans_ixs = numerical_ix
-            transformers.append(("cats", "passthrough", categorical_features))
-            cat_ix = list(range(len(categorical_features)))  # They are at start
-
-        else:
-            raise ValueError(
-                f"Unrecognized combination of {self.apply_to_categorical=}"
-                f" and {self.append_to_original=}",
-            )
-
-        # NOTE: No need to keep track of categoricals here, already done above
-        if self.transform_name != "per_feature":
-            _transformer = all_preprocessors[self.transform_name]
-            transformers.append(("feat_transform", _transformer, trans_ixs))
-        else:
-            preprocessors = list(all_preprocessors.values())
-            transformers.extend(
-                [
-                    (f"transformer_{i}", rng.choice(preprocessors), [i])  # type: ignore
-                    for i in trans_ixs
-                ],
-            )
-
-        transformer = ColumnTransformer(
-            transformers,
-            remainder="drop",
-            sparse_threshold=0.0,  # No sparse
-        )
-
-        # Apply a global transformer which accepts the entire dataset instead of
-        # one column
-        # NOTE: We assume global_transformer does not destroy the semantic meaning of
-        # categorical_features_.
-        if global_transformer_:
-            transformer = Pipeline(
-                [
-                    ("preprocess", transformer),
-                    ("global_transformer", global_transformer_),
-                ],
-            )
-
-        self.transformer_ = transformer
-
-        return transformer, cat_ix
-
-    @override
-    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
-        n_samples, n_features = X.shape
-        transformer, cat_ix = self._set_transformer_and_cat_ix(
-            n_samples,
-            n_features,
-            categorical_features,
-        )
-        transformer.fit(X[:, self.subsampled_features_])
-        self.categorical_features_after_transform_ = cat_ix
-        self.transformer_ = transformer
-        return cat_ix
-
-    @override
-    def fit_transform(
-        self,
-        X: np.ndarray,
-        categorical_features: list[int],
-    ) -> TransformResult:
-        n_samples, n_features = X.shape
-        transformer, cat_ix = self._set_transformer_and_cat_ix(
-            n_samples,
-            n_features,
-            categorical_features,
-        )
-        Xt = transformer.fit_transform(X[:, self.subsampled_features_])
-        self.categorical_features_after_transform_ = cat_ix
-        self.transformer_ = transformer
-        return TransformResult(Xt, cat_ix)  # type: ignore
-
-    @override
-    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
-        assert self.transformer_ is not None, "You must call fit first"
-        return self.transformer_.transform(X[:, self.subsampled_features_])  # type: ignore
-
-
-def get_all_global_transformers(
-    num_examples: int,
-    num_features: int,
-    random_state: int | None = None,
-) -> dict[str, FeatureUnion | Pipeline]:
-    """Returns a dictionary of global transformers to transform the data."""
-    return {
-        "scaler": _make_standard_scaler_safe(("standard", StandardScaler())),
-        "svd": FeatureUnion(
-            [
-                # default FunctionTransformer yields the identity function
-                ("passthrough", FunctionTransformer()),
-                (
-                    "svd",
-                    Pipeline(
-                        steps=[
-                            (
-                                "save_standard",
-                                _make_standard_scaler_safe(
-                                    ("standard", StandardScaler(with_mean=False)),
-                                ),
-                            ),
-                            (
-                                "svd",
-                                TruncatedSVD(
-                                    algorithm="arpack",
-                                    n_components=max(
-                                        1,
-                                        min(
-                                            num_examples // 10 + 1,
-                                            num_features // 2,
-                                        ),
-                                    ),
-                                    random_state=random_state,
-                                ),
-                            ),
-                        ],
-                    ),
-                ),
-            ],
-        ),
-        "svd_quarter_components": FeatureUnion(
-            [
-                ("passthrough", FunctionTransformer(func=_identity)),
-                (
-                    "svd",
-                    Pipeline(
-                        steps=[
-                            (
-                                "save_standard",
-                                _make_standard_scaler_safe(
-                                    ("standard", StandardScaler(with_mean=False)),
-                                ),
-                            ),
-                            (
-                                "svd",
-                                TruncatedSVD(
-                                    algorithm="arpack",
-                                    n_components=max(
-                                        1,
-                                        min(
-                                            num_examples // 10 + 1,
-                                            num_features // 4,
-                                        ),
-                                    ),
-                                    random_state=random_state,
-                                ),
-                            ),
-                        ],
-                    ),
-                ),
-            ],
-        ),
-    }
-
-
-def get_adaptive_preprocessors(
-    num_examples: int = 100,
-    random_state: int | None = None,
-) -> dict[str, ColumnTransformer]:
-    """Returns a dictionary of adaptive column transformers that can be used to
-    preprocess the data. Adaptive column transformers are used to preprocess the
-    data based on the column type, they receive a pandas dataframe with column
-    names, that indicate the column type. Column types are not datatypes,
-    but rather a string that indicates how the data should be preprocessed.
-
-    Args:
-        num_examples: The number of examples in the dataset.
-        random_state: The random state to use for the transformers.
-    """
-    return {
-        "adaptive": ColumnTransformer(
-            [
-                (
-                    "skewed_pos_1_0",
-                    FunctionTransformer(
-                        func=np.exp,
-                        inverse_func=np.log,
-                        check_inverse=False,
-                    ),
-                    make_column_selector("skewed_pos_1_0*"),
-                ),
-                (
-                    "skewed_pos",
-                    _make_box_cox_safe(
-                        _add_safe_standard_to_safe_power_without_standard(
-                            SafePowerTransformer(
-                                standardize=False,
-                                method="box-cox",
-                            ),
-                        ),
-                    ),
-                    make_column_selector("skewed_pos*"),
-                ),
-                (
-                    "skewed",
-                    _add_safe_standard_to_safe_power_without_standard(
-                        SafePowerTransformer(
-                            standardize=False,
-                            method="yeo-johnson",
-                        ),
-                    ),
-                    make_column_selector("skewed*"),
-                ),
-                (
-                    "other",
-                    AdaptiveQuantileTransformer(
-                        output_distribution="normal",
-                        n_quantiles=max(num_examples // 10, 2),
-                        random_state=random_state,
-                    ),
-                    # "other" or "ordinal"
-                    make_column_selector("other*"),
-                ),
-                (
-                    "ordinal",
-                    # default FunctionTransformer yields the identity function
-                    FunctionTransformer(),
-                    # "other" or "ordinal"
-                    make_column_selector("ordinal*"),
-                ),
-                (
-                    "normal",
-                    # default FunctionTransformer yields the identity function
-                    FunctionTransformer(),
-                    make_column_selector("normal*"),
-                ),
-            ],
-            remainder="passthrough",
-        ),
-    }
-
-
-def get_all_reshape_feature_distribution_preprocessors(
-    num_examples: int,
-    random_state: int | None = None,
-) -> dict[str, TransformerMixin | Pipeline]:
-    """Returns a dictionary of preprocessors to preprocess the data."""
-    all_preprocessors = {
-        "power": _add_safe_standard_to_safe_power_without_standard(
-            PowerTransformer(standardize=False),
-        ),
-        "safepower": _add_safe_standard_to_safe_power_without_standard(
-            SafePowerTransformer(standardize=False),
-        ),
-        "power_box": _make_box_cox_safe(
-            _add_safe_standard_to_safe_power_without_standard(
-                PowerTransformer(standardize=False, method="box-cox"),
-            ),
-        ),
-        "safepower_box": _make_box_cox_safe(
-            _add_safe_standard_to_safe_power_without_standard(
-                SafePowerTransformer(standardize=False, method="box-cox"),
-            ),
-        ),
-        "log": FunctionTransformer(
-            func=np.log,
-            inverse_func=np.exp,
-            check_inverse=False,
-        ),
-        "1_plus_log": FunctionTransformer(
-            func=np.log1p,
-            inverse_func=_exp_minus_1,
-            check_inverse=False,
-        ),
-        "exp": FunctionTransformer(
-            func=np.exp,
-            inverse_func=np.log,
-            check_inverse=False,
-        ),
-        "quantile_uni_coarse": AdaptiveQuantileTransformer(
-            output_distribution="uniform",
-            n_quantiles=max(num_examples // 10, 2),
-            random_state=random_state,
-        ),
-        "quantile_norm_coarse": AdaptiveQuantileTransformer(
-            output_distribution="normal",
-            n_quantiles=max(num_examples // 10, 2),
-            random_state=random_state,
-        ),
-        "quantile_uni": AdaptiveQuantileTransformer(
-            output_distribution="uniform",
-            n_quantiles=max(num_examples // 5, 2),
-            random_state=random_state,
-        ),
-        "quantile_norm": AdaptiveQuantileTransformer(
-            output_distribution="normal",
-            n_quantiles=max(num_examples // 5, 2),
-            random_state=random_state,
-        ),
-        "quantile_uni_fine": AdaptiveQuantileTransformer(
-            output_distribution="uniform",
-            n_quantiles=num_examples,
-            random_state=random_state,
-        ),
-        "quantile_norm_fine": AdaptiveQuantileTransformer(
-            output_distribution="normal",
-            n_quantiles=num_examples,
-            random_state=random_state,
-        ),
-        "squashing_scaler_default": SquashingScaler(),
-        "squashing_scaler_max10": SquashingScaler(max_absolute_value=10.0),
-        "robust": RobustScaler(unit_variance=True),
-        # default FunctionTransformer yields the identity function
-        "none": FunctionTransformer(),
-        **get_all_kdi_transformers(),
-    }
-
-    with contextlib.suppress(Exception):
-        all_preprocessors["norm_and_kdi"] = FeatureUnion(
-            [
-                (
-                    "norm",
-                    AdaptiveQuantileTransformer(
-                        output_distribution="normal",
-                        n_quantiles=max(num_examples // 10, 2),
-                        random_state=random_state,
-                    ),
-                ),
-                (
-                    "kdi",
-                    KDITransformerWithNaN(alpha=1.0, output_distribution="uniform"),
-                ),
-            ],
-        )
-
-    all_preprocessors.update(
-        get_adaptive_preprocessors(
-            num_examples,
-            random_state=random_state,
-        ),
-    )
-
-    return all_preprocessors
-
-
-__all__ = [
-    "ReshapeFeatureDistributionsStep",
-    "get_all_reshape_feature_distribution_preprocessors",
-]
diff --git a/src/tabpfn/preprocessors/safe_power_transformer.py b/src/tabpfn/preprocessors/safe_power_transformer.py
deleted file mode 100644
index 85653f5..0000000
--- a/src/tabpfn/preprocessors/safe_power_transformer.py
+++ /dev/null
@@ -1,200 +0,0 @@
-"""Safe Power Transformer."""
-
-from __future__ import annotations
-
-from typing_extensions import override
-
-import numpy as np
-from scipy import optimize
-from sklearn.preprocessing import PowerTransformer
-
-
-# this is taken from https://github.com/scipy/scipy/pull/18852
-# which fix overflow issues
-# we can directly import from scipy once we drop support for scipy < 1.12
-def _yeojohnson(
-    x: np.ndarray,
-    lmbda: float | None = None,
-) -> tuple[np.ndarray, float | None]:
-    x = np.asarray(x)
-    if x.size == 0:
-        # changed from scipy from return x
-        return (x, None) if lmbda is None else x  # type: ignore
-
-    if np.issubdtype(x.dtype, np.complexfloating):
-        raise ValueError(
-            "Yeo-Johnson transformation is not defined for complex numbers."
-        )
-
-    if np.issubdtype(x.dtype, np.integer):
-        x = x.astype(np.float64, copy=False)
-
-    if lmbda is not None:
-        return _yeojohnson_transform(x, lmbda)  # type: ignore
-
-    # if lmbda=None, find the lmbda that maximizes the log-likelihood function.
-    lmax = _yeojohnson_normmax(x)
-    y = _yeojohnson_transform(x, lmax)
-
-    return y, lmax
-
-
-def _yeojohnson_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
-    """Returns `x` transformed by the Yeo-Johnson power transform with given
-    parameter `lmbda`.
-    """
-    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
-    out = np.zeros_like(x, dtype=dtype)
-    pos = x >= 0  # binary mask
-
-    # when x >= 0
-    if abs(lmbda) < np.spacing(1.0):
-        out[pos] = np.log1p(x[pos])
-    else:  # lmbda != 0
-        # more stable version of: ((x + 1) ** lmbda - 1) / lmbda
-        out[pos] = np.expm1(lmbda * np.log1p(x[pos])) / lmbda
-
-    # when x < 0
-    if abs(lmbda - 2) > np.spacing(1.0):
-        out[~pos] = -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda)
-    else:  # lmbda == 2
-        out[~pos] = -np.log1p(-x[~pos])
-
-    return out
-
-
-def _yeojohnson_llf(lmb: float, data: np.ndarray) -> np.ndarray:
-    r"""The yeojohnson log-likelihood function."""
-    data = np.asarray(data)
-    n_samples = data.shape[0]
-
-    if n_samples == 0:
-        return np.nan  # type: ignore
-
-    trans = _yeojohnson_transform(data, lmb)
-    trans_var = trans.var(axis=0)
-    loglike = np.empty_like(trans_var)
-
-    # Avoid RuntimeWarning raised by np.log when the variance is too low
-    tiny_variance = trans_var < np.finfo(trans_var.dtype).tiny
-    loglike[tiny_variance] = np.inf
-
-    loglike[~tiny_variance] = -n_samples / 2 * np.log(trans_var[~tiny_variance])
-    loglike[~tiny_variance] += (lmb - 1) * (np.sign(data) * np.log1p(np.abs(data))).sum(
-        axis=0
-    )
-    return loglike
-
-
-def _yeojohnson_normmax(x: np.ndarray, brack: float | None = None) -> float:
-    """Compute optimal Yeo-Johnson transform parameter.
-    Compute optimal Yeo-Johnson transform parameter for input data, using
-    maximum likelihood estimation.
-
-    """
-
-    def _neg_llf(lmbda: float, data: np.ndarray) -> np.ndarray:
-        llf = _yeojohnson_llf(lmbda, data)
-        # reject likelihoods that are inf which are likely due to small
-        # variance in the transformed space
-        llf[np.isinf(llf)] = -np.inf
-        return -llf
-
-    with np.errstate(invalid="ignore"):
-        if not np.all(np.isfinite(x)):
-            raise ValueError("Yeo-Johnson input must be finite.")
-        if np.all(x == 0):
-            return 1.0
-        if brack is not None:
-            return optimize.brent(_neg_llf, brack=brack, args=(x,))  # type: ignore
-        x = np.asarray(x)
-        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
-        # Allow values up to 20 times the maximum observed value to be safely
-        # transformed without over- or underflow.
-        log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
-        # Use half of floating point's exponent range to allow safe computation
-        # of the variance of the transformed data.
-        log_eps = np.log(np.finfo(dtype).eps)
-        log_tiny_float = (np.log(np.finfo(dtype).tiny) - log_eps) / 2
-        log_max_float = (np.log(np.finfo(dtype).max) + log_eps) / 2
-        # Compute the bounds by approximating the inverse of the Yeo-Johnson
-        # transform on the smallest and largest floating point exponents, given
-        # the largest data we expect to observe. See [1] for further details.
-        # [1] https://github.com/scipy/scipy/pull/18852#issuecomment-1630286174
-        lb = log_tiny_float / log1p_max_x
-        ub = log_max_float / log1p_max_x
-        # Convert the bounds if all or some of the data is negative.
-        if np.all(x < 0):
-            lb, ub = 2 - ub, 2 - lb
-        elif np.any(x < 0):
-            lb, ub = max(2 - ub, lb), min(2 - lb, ub)
-        # Match `optimize.brent`'s tolerance.
-        tol_brent = 1.48e-08
-        return optimize.fminbound(_neg_llf, lb, ub, args=(x,), xtol=tol_brent)  # type: ignore
-
-
-# we created this inspired by the scipy change for transform
-# https://github.com/scipy/scipy/pull/18852
-# this is not in scipy even 1.12
-def _yeojohnson_inverse_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
-    """Return inverse-transformed input x following Yeo-Johnson inverse
-    transform with parameter lambda.
-    """
-    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
-    x_inv = np.zeros_like(x, dtype=dtype)
-    pos = x >= 0
-
-    # when x >= 0
-    if abs(lmbda) < np.spacing(1.0):
-        x_inv[pos] = np.expm1(x[pos])
-    else:  # lmbda != 0
-        # more stable version of: (x * lmbda + 1) ** (1 / lmbda) - 1
-        x_inv[pos] = np.expm1(np.log(x[pos] * lmbda + 1) / lmbda)
-
-    # when x < 0
-    if abs(lmbda - 2) > np.spacing(1.0):
-        # more stable version of: 1 - (-(2 - lmbda) * x + 1) ** (1 / (2 - lmbda))
-        x_inv[~pos] = -np.expm1(np.log(-(2 - lmbda) * x[~pos] + 1) / (2 - lmbda))
-    else:  # lmbda == 2
-        x_inv[~pos] = -np.expm1(-x[~pos])
-
-    return x_inv
-
-
-class SafePowerTransformer(PowerTransformer):
-    """Variant of PowerTransformer that uses the scipy yeo-johnson functions
-    which have been fixed to avoid overflow issues.
-    """
-
-    def __init__(
-        self,
-        method: str = "yeo-johnson",
-        *,
-        standardize: bool = True,
-        copy: bool = True,
-    ) -> None:
-        super().__init__(method=method, standardize=standardize, copy=copy)
-
-    # requires scipy >= 1.9
-    # this is the default in scikit-learn main for versions > 1.7
-    # see https://github.com/scikit-learn/scikit-learn/pull/31227
-    @override
-    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
-        # the computation of lambda is influenced by NaNs so we need to
-        # get rid of them
-        x = x[~np.isnan(x)]
-        _, lmbda = _yeojohnson(x, lmbda=None)
-        return lmbda  # type: ignore
-
-    @override
-    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
-        return _yeojohnson_transform(x, lmbda)
-
-    @override
-    def _yeo_johnson_inverse_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
-        return _yeojohnson_inverse_transform(x, lmbda)
-
-
-__all__ = [
-    "SafePowerTransformer",
-]
diff --git a/src/tabpfn/preprocessors/shuffle_features_step.py b/src/tabpfn/preprocessors/shuffle_features_step.py
deleted file mode 100644
index 6e8abdd..0000000
--- a/src/tabpfn/preprocessors/shuffle_features_step.py
+++ /dev/null
@@ -1,68 +0,0 @@
-"""Shuffle Features Step."""
-
-from __future__ import annotations
-
-from typing import Literal
-from typing_extensions import override
-
-import numpy as np
-import torch
-
-from tabpfn.preprocessors.preprocessing_helpers import (
-    FeaturePreprocessingTransformerStep,
-)
-from tabpfn.utils import infer_random_state
-
-
-class ShuffleFeaturesStep(FeaturePreprocessingTransformerStep):
-    """Shuffle the features in the data."""
-
-    def __init__(
-        self,
-        shuffle_method: Literal["shuffle", "rotate"] | None = "rotate",
-        shuffle_index: int = 0,
-        random_state: int | np.random.Generator | None = None,
-    ):
-        super().__init__()
-        self.random_state = random_state
-        self.shuffle_method = shuffle_method
-        self.shuffle_index = shuffle_index
-
-        self.index_permutation_: list[int] | torch.Tensor | None = None
-
-    @override
-    def _fit(
-        self,
-        X: np.ndarray | torch.Tensor,
-        categorical_features: list[int],
-    ) -> list[int]:
-        _, rng = infer_random_state(self.random_state)
-        if self.shuffle_method == "rotate":
-            index_permutation = np.roll(
-                np.arange(X.shape[1]),
-                self.shuffle_index,
-            ).tolist()
-        elif self.shuffle_method == "shuffle":
-            index_permutation = rng.permutation(X.shape[1]).tolist()
-        elif self.shuffle_method is None:
-            index_permutation = np.arange(X.shape[1]).tolist()
-        else:
-            raise ValueError(f"Unknown shuffle method {self.shuffle_method}")
-        if isinstance(X, torch.Tensor):
-            self.index_permutation_ = torch.tensor(index_permutation, dtype=torch.long)
-        else:
-            self.index_permutation_ = index_permutation
-
-        return [
-            new_idx
-            for new_idx, idx in enumerate(index_permutation)
-            if idx in categorical_features
-        ]
-
-    @override
-    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
-        assert self.index_permutation_ is not None, "You must call fit first"
-        assert len(self.index_permutation_) == X.shape[1], (
-            "The number of features must not change after fit"
-        )
-        return X[:, self.index_permutation_]
diff --git a/src/tabpfn/preprocessors/squashing_scaler_transformer.py b/src/tabpfn/preprocessors/squashing_scaler_transformer.py
deleted file mode 100644
index dccdab3..0000000
--- a/src/tabpfn/preprocessors/squashing_scaler_transformer.py
+++ /dev/null
@@ -1,389 +0,0 @@
-"""Implementation of the SquashingScaler, adapted from skrub.
-
-See https://skrub-data.org/stable/reference/generated/skrub.SquashingScaler.html
-
-This preprocessing is used e.g. in RealMLP, see https://arxiv.org/abs/2407.04491
-"""
-
-from __future__ import annotations
-
-import numbers
-from typing import Any
-from typing_extensions import override
-
-import numpy as np
-from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
-from sklearn.preprocessing import RobustScaler
-from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
-
-try:
-    from sklearn.utils.validation import validate_data as sklearn_validate_data
-except ImportError:
-    sklearn_validate_data = None
-
-
-def _validate_data(estimator: BaseEstimator, **kwargs: Any) -> Any:
-    """Select the appropriate validate_data API and runs it."""
-    if sklearn_validate_data is not None:
-        return sklearn_validate_data(estimator, **kwargs)
-
-    if "ensure_all_finite" in kwargs:
-        force_all_finite = kwargs.pop("ensure_all_finite")
-    else:
-        force_all_finite = True
-    return estimator._validate_data(**kwargs, force_all_finite=force_all_finite)
-
-
-def _mask_inf(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
-    """Replace infinite values with NaN and return their sign."""
-    if (mask_inf := np.isinf(X)).any():
-        sign = np.sign(X)
-        X = np.where(mask_inf, np.nan, X)
-        # 0 when X is finite, 1 when X is +inf, -1 when X is -inf
-        mask_inf = mask_inf.astype(X.dtype) * sign
-
-    return X, mask_inf
-
-
-def _set_zeros(X: np.ndarray, zero_cols: np.ndarray) -> np.ndarray:
-    """Set the finite values of the specified columns to zero."""
-    mask = np.isfinite(X)
-    mask[:, ~zero_cols] = False
-    X[mask] = 0.0
-    return X
-
-
-def _soft_clip(
-    X: np.ndarray,
-    max_absolute_value: float,
-    mask_inf: np.ndarray,
-) -> np.ndarray:
-    """Apply a soft clipping to the data.
-
-    Parameters
-    ----------
-    X : array-like, shape (n_samples, n_features)
-        The data to be clipped.
-    max_absolute_value : float, default=3.0
-        Maximum absolute value that the transformed data can take.
-    mask_inf : array-like, shape (n_samples, n_features)
-        A mask indicating the positions of infinite values in the input data and their
-        signs.
-
-    Returns:
-    -------
-    X_clipped : array-like, shape (n_samples, n_features)
-        The clipped version of the input.
-    """
-    X = X / np.sqrt(1 + (X / max_absolute_value) ** 2)
-    X = np.where(mask_inf == 1, max_absolute_value, X)
-    return np.where(mask_inf == -1, -max_absolute_value, X)
-
-
-class _MinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
-    """A variation of scikit-learn MinMaxScaler.
-
-    A simple min-max scaler that centers the median to zero and scales
-    the data to the range [-1, 1].
-
-    scikit-learn MinMaxScaler computes the following::
-
-        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
-        X_scaled = X_std * (max - min) + min
-
-    This scaler computes the following::
-
-        X_scaled = 2 * (X - median) / (X.max(axis=0) - X.min(axis=0) + eps)
-    """
-
-    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> _MinMaxScaler:
-        del y
-        eps = np.finfo("float32").tiny
-        self.median_ = np.nanmedian(X, axis=0)
-        self.scale_ = 2 / (np.nanmax(X, axis=0) - np.nanmin(X, axis=0) + eps)
-        return self
-
-    def transform(self, X: np.ndarray) -> np.ndarray:
-        check_is_fitted(self, ["median_", "scale_"])
-        return self.scale_ * (X - self.median_)
-
-
-class SquashingScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
-    r"""Perform robust centering and scaling followed by soft clipping.
-
-    When features have large outliers, smooth clipping prevents the outliers from
-    affecting the result too strongly, while robust scaling prevents the outliers from
-    affecting the inlier scaling. Infinite values are mapped to the corresponding
-    boundaries of the interval. NaN values are preserved.
-
-    Parameters
-    ----------
-    max_absolute_value : float, default=3.0
-        Maximum absolute value that the transformed data can take.
-
-    quantile_range : tuple of float, default=(0.25, 0.75)
-        The quantiles used to compute the scaling factor. The first value is the lower
-        quantile and the second value is the upper quantile. The default values are the
-        25th and 75th percentiles, respectively. The quantiles are used to compute the
-        scaling factor for the robust scaling step. The quantiles are computed from the
-        finite values in the input column. If the two quantiles are equal, the scaling
-        factor is computed from the 0th and 100th percentiles (i.e., the minimum and
-        maximum values of the finite values in the input column).
-
-    Notes:
-    -----
-    This transformer is applied to each column independently. It uses two stages:
-
-    1. The first stage centers the median of the data to zero and multiplies the data by a
-       scaling factor determined from quantiles of the distribution, using
-       scikit-learn's :class:`~sklearn.preprocessing.RobustScaler`. It also handles
-       edge-cases in which the two quantiles are equal by following-up with a
-       :class:`~sklearn.preprocessing.MinMaxScaler`.
-    2. The second stage applies a soft clipping to the transformed data to limit the
-       data to the interval ``[-max_absolute_value, max_absolute_value]`` in an
-       injective way.
-
-    Infinite values will be mapped to the corresponding boundaries of the interval. NaN
-    values will be preserved.
-
-    The formula for the transform is:
-
-    .. math::
-
-        \begin{align*}
-            a &:= \begin{cases}
-                1/(q_{\beta} - q_{\alpha}) &\text{if} \quad q_{\beta} \neq q_{\alpha} \\
-                2/(q_1 - q_0) &\text{if}\quad q_{\beta} = q_{\alpha} \text{ and } q_1
-                \neq q_0 \\ 0 & \text{otherwise}
-            \end{cases} \\ z &:= a.(x - q_{1/2}), \\ x_{\text{out}} &:= \frac{z}{\sqrt{1
-            + (z/B)^2}},
-        \end{align*}
-
-    where:
-
-    - :math:`x` is a value in the input column.
-    - :math:`q_{\gamma}` is the :math:`\gamma`-quantile of the finite values in X,
-    - :math:`B` is max_abs_value
-    - :math:`\alpha` is the lower quantile
-    - :math:`\beta` is the upper quantile.
-
-    References:
-    ----------
-    This method has been introduced as the robust scaling and smooth clipping transform
-    in `Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data
-    (Holzm├╝ller et al., 2024) <https://arxiv.org/abs/2407.04491>`_.
-
-    Examples:
-    --------
-    >>> import pandas as pd
-    >>> import numpy as np
-    >>> from tabpfn.preprocessors import SquashingScaler
-
-    In the general case, this scale uses a RobustScaler:
-
-    >>> X = pd.DataFrame(dict(col=[np.inf, -np.inf, 3, -1, np.nan, 2]))
-    >>> SquashingScaler(max_absolute_value=3).fit_transform(X)
-    array([[ 3.        ],
-           [-3.        ],
-           [ 0.49319696],
-           [-1.34164079],
-           [        nan],
-           [ 0.        ]])
-
-    When quantile ranges are equal, this scaler uses a customized MinMaxScaler:
-
-    >>> X = pd.DataFrame(dict(col=[0, 1, 1, 1, 2, np.nan]))
-    >>> SquashingScaler().fit_transform(X)
-    array([[-0.9486833],
-           [ 0.       ],
-           [ 0.       ],
-           [ 0.       ],
-           [ 0.9486833],
-           [       nan]])
-
-    Finally, when the min and max are equal, this scaler fills the column with zeros:
-
-    >>> X = pd.DataFrame(dict(col=[1, 1, 1, np.nan]))
-    >>> SquashingScaler().fit_transform(X)
-    array([[ 0.],
-           [ 0.],
-           [ 0.],
-           [nan]])
-    """  # noqa: E501
-
-    robust_scaler_: RobustScaler | None
-    minmax_scaler_: _MinMaxScaler | None
-    robust_cols_: np.ndarray
-    minmax_cols_: np.ndarray
-    zero_cols_: np.ndarray
-
-    def __init__(
-        self,
-        max_absolute_value: float = 3.0,
-        quantile_range: tuple[float, float] = (25.0, 75.0),
-    ) -> None:
-        super().__init__()
-        self.max_absolute_value = max_absolute_value
-        self.quantile_range = quantile_range
-
-    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> SquashingScaler:
-        """Fit the transformer to a column.
-
-        Parameters
-        ----------
-        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
-            The data to transform.
-        y : None
-            Unused. Here for compatibility with scikit-learn.
-
-        Returns:
-        -------
-        self : SquashingScaler
-            The fitted transformer.
-        """
-        del y
-
-        if not (
-            isinstance(self.max_absolute_value, numbers.Number)
-            and np.isfinite(self.max_absolute_value)
-            and self.max_absolute_value > 0
-        ):
-            raise ValueError(
-                f"Got max_absolute_value={self.max_absolute_value!r}, but expected a "
-                "positive finite number."
-            )
-
-        X = _validate_data(
-            self,
-            X=X,  # type: ignore
-            reset=True,
-            dtype=FLOAT_DTYPES,
-            accept_sparse=False,
-            ensure_2d=True,
-            ensure_all_finite=False,
-        )
-        # To use sklearn scalers, we need to convert np.inf to np.nan.
-        X, _ = _mask_inf(X)
-
-        # For each column, we apply 1 out of 3 scaling methods:
-        # If the max is equal to the min, then we fill the column with zeros.
-        zero_cols = np.nanmax(X, axis=0) == np.nanmin(X, axis=0)
-
-        # If the two quantiles defined by quantile_range have the same values, we
-        # use a customized MinMaxScaler. We remove from this selection columns that
-        # are already selected as zero cols (i.e. columns that have the same min and max
-        # also have the same quantile_range values).
-        quantiles = np.nanpercentile(X, self.quantile_range, axis=0)
-        minmax_cols = quantiles[0, :] == quantiles[1, :]
-        minmax_cols = minmax_cols & ~zero_cols
-
-        # Otherwise (general case), we use a RobustScaler.
-        robust_cols = ~(minmax_cols | zero_cols)
-
-        if robust_cols.any():
-            self.robust_scaler_ = RobustScaler(
-                with_centering=True,
-                with_scaling=True,
-                quantile_range=self.quantile_range,
-                copy=True,
-            )
-            self.robust_scaler_ = self.robust_scaler_.fit(X[:, robust_cols])
-        else:
-            self.robust_scaler_ = None
-        self.robust_cols_ = robust_cols
-
-        if minmax_cols.any():
-            self.minmax_scaler_ = _MinMaxScaler()
-            self.minmax_scaler_ = self.minmax_scaler_.fit(X[:, minmax_cols])
-        else:
-            self.minmax_scaler_ = None
-        self.minmax_cols_ = minmax_cols
-
-        self.zero_cols_ = zero_cols
-
-        return self
-
-    @override
-    def fit_transform(
-        self,
-        X: np.ndarray,
-        y: None | np.ndarray = None,
-        **fit_params: Any,
-    ) -> np.ndarray:
-        """Fit the transformer and transform a column.
-
-        Parameters
-        ----------
-        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
-            The data to transform.
-        y : None
-            Unused. Here for compatibility with scikit-learn.
-
-        Returns:
-        -------
-        X_out: numpy array, shape (n_samples, n_features)
-            The transformed version of the input.
-        """
-        del y
-        del fit_params
-
-        self.fit(X)
-        return self.transform(X)
-
-    def transform(self, X: np.ndarray) -> np.ndarray:
-        """Transform a column.
-
-        Parameters
-        ----------
-        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
-            The data to transform.
-
-        Returns:
-        -------
-        X_out: numpy array of shape (n_samples, n_features)
-            The transformed version of the input.
-        """
-        check_is_fitted(
-            self,
-            [
-                "robust_scaler_",
-                "minmax_scaler_",
-                "zero_cols_",
-                "robust_cols_",
-                "minmax_cols_",
-            ],
-        )
-
-        X = _validate_data(
-            self,
-            X=X,  # type: ignore
-            reset=False,
-            dtype=FLOAT_DTYPES,
-            accept_sparse=False,
-            ensure_2d=True,
-            ensure_all_finite=False,
-        )
-        # To replace the original ┬▒np.inf with ┬▒max_absolute_value in the final output.
-        # mask_inf is a 2D array containing the sign of the np.inf in the input.
-        X, mask_inf = _mask_inf(X)
-
-        # copy the input since we change the values in place
-        X_tr = X.copy()
-        if self.robust_cols_.any():
-            assert self.robust_scaler_ is not None
-            X_tr[:, self.robust_cols_] = self.robust_scaler_.transform(
-                X[:, self.robust_cols_]
-            )
-        if self.minmax_cols_.any():
-            assert self.minmax_scaler_ is not None
-            X_tr[:, self.minmax_cols_] = self.minmax_scaler_.transform(
-                X[:, self.minmax_cols_]
-            )
-        if self.zero_cols_.any():
-            # if the scale is 0, we set the values to 0
-            X_tr = _set_zeros(X_tr, self.zero_cols_)
-
-        return _soft_clip(X_tr, self.max_absolute_value, mask_inf)
-
-
-__all__ = ["SquashingScaler"]
diff --git a/src/tabpfn/regressor.py b/src/tabpfn/regressor.py
index c53b01a..7fdca58 100644
--- a/src/tabpfn/regressor.py
+++ b/src/tabpfn/regressor.py
@@ -19,12 +19,11 @@ from __future__ import annotations
 
 import logging
 import typing
-import warnings
 from collections.abc import Callable, Sequence
 from functools import partial
 from pathlib import Path
-from typing import TYPE_CHECKING, Annotated, Any, Literal, Union
-from typing_extensions import Self, TypedDict, deprecated, overload
+from typing import TYPE_CHECKING, Any, Literal, Union
+from typing_extensions import Self, TypedDict, overload
 
 import numpy as np
 import torch
@@ -35,41 +34,34 @@ from sklearn.base import (
     TransformerMixin,
     check_is_fitted,
 )
-from tabpfn_common_utils.telemetry import track_model_call
-from tabpfn_common_utils.telemetry.interactive import ping
 
 from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
 from tabpfn.base import (
     RegressorModelSpecs,
+    _initialize_model_variables_helper,
     check_cpu_warning,
     create_inference_engine,
     determine_precision,
     get_preprocessed_datasets_helper,
-    initialize_model_variables_helper,
 )
-from tabpfn.constants import REGRESSION_CONSTANT_TARGET_BORDER_EPSILON, ModelVersion
 from tabpfn.inference import InferenceEngine, InferenceEngineBatchedNoPreprocessing
-from tabpfn.model_loading import (
-    ModelSource,
-    get_cache_dir,
-    load_fitted_tabpfn_model,
-    save_fitted_tabpfn_model,
-)
+from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
 from tabpfn.preprocessing import (
     DatasetCollectionWithPreprocessing,
     EnsembleConfig,
+    PreprocessorConfig,
     RegressorEnsembleConfig,
+    ReshapeFeatureDistributionsStep,
+    default_regressor_preprocessor_configs,
 )
-from tabpfn.preprocessors import get_all_reshape_feature_distribution_preprocessors
-from tabpfn.preprocessors.preprocessing_helpers import get_ordinal_encoder
 from tabpfn.utils import (
-    DevicesSpecification,
-    fix_dtypes,
-    get_embeddings,
+    _fix_dtypes,
+    _get_embeddings,
+    _get_ordinal_encoder,
+    _process_text_na_dataframe,
+    _transform_borders_one,
     infer_categorical_features,
     infer_random_state,
-    process_text_na_dataframe,
-    transform_borders_one,
     translate_probs_across_borders,
     validate_X_predict,
     validate_Xy_fit,
@@ -81,11 +73,10 @@ if TYPE_CHECKING:
     from sklearn.pipeline import Pipeline
     from torch.types import _dtype
 
-    from tabpfn.architectures.base.memory import MemorySavingMode
-    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
+    from tabpfn.architectures.interface import ArchitectureConfig
+    from tabpfn.config import ModelInterfaceConfig
     from tabpfn.constants import XType, YType
     from tabpfn.inference import InferenceEngine
-    from tabpfn.inference_config import InferenceConfig
 
     try:
         from sklearn.base import Tags
@@ -134,29 +125,18 @@ RegressionResultType = Union[
 class TabPFNRegressor(RegressorMixin, BaseEstimator):
     """TabPFNRegressor class."""
 
-    configs_: list[ArchitectureConfig]
-    """The configurations of the loaded models to be used for inference.
-
-    The concrete type of these configs is defined by the architectures in use and should
-    be inspected at runtime, but they will be subclasses of ArchitectureConfig.
-    """
-
-    models_: list[Architecture]
-    """The loaded models to be used for inference.
+    config_: ArchitectureConfig
+    """The configuration of the loaded model to be used for inference.
 
-    The models can be different PyTorch modules, but will be subclasses of Architecture.
+    The concrete type of this config is defined by the arhitecture in use and should be
+    inspected at runtime, but it will be a subclass of ArchitectureConfig.
     """
 
-    inference_config_: InferenceConfig
-    """Additional configuration of inference for expert users."""
+    interface_config_: ModelInterfaceConfig
+    """Additional configuration of the interface for expert users."""
 
-    devices_: tuple[torch.device, ...]
-    """The devices determined to be used.
-
-    The devices are determined based on the `device` argument to the constructor, and
-    the devices available on the system. If multiple devices are listed, currently only
-    the first is used for inference.
-    """
+    device_: torch.device
+    """The device determined to be used."""
 
     feature_names_in_: npt.NDArray[Any]
     """The feature names of the input data.
@@ -177,14 +157,11 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
     n_outputs_: Literal[1]  # We only support single output
     """The number of outputs the model supports. Only 1 for now"""
 
-    znorm_space_bardist_: FullSupportBarDistribution
-    """The bar distribution of the target variable, used by the model.
-    This is the bar distribution in the normalized target space.
-    """
+    bardist_: FullSupportBarDistribution
+    """The bar distribution of the target variable, used by the model."""
 
-    raw_space_bardist_: FullSupportBarDistribution
-    """The bar distribution in the raw target space, used for computing the
-    predictions."""
+    normalized_bardist_: FullSupportBarDistribution
+    """The normalized bar distribution used for computing the predictions."""
 
     use_autocast_: bool
     """Whether torch's autocast should be used."""
@@ -205,14 +182,8 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         categorical_features_indices: Sequence[int] | None = None,
         softmax_temperature: float = 0.9,
         average_before_softmax: bool = False,
-        model_path: str
-        | Path
-        | list[str]
-        | list[Path]
-        | Literal["auto"]
-        | RegressorModelSpecs
-        | list[RegressorModelSpecs] = "auto",
-        device: DevicesSpecification = "auto",
+        model_path: str | Path | Literal["auto"] | RegressorModelSpecs = "auto",
+        device: str | torch.device | Literal["auto"] = "auto",
         ignore_pretraining_limits: bool = False,
         inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
         fit_mode: Literal[
@@ -221,18 +192,13 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             "fit_with_cache",
             "batched",
         ] = "fit_preprocessors",
-        memory_saving_mode: MemorySavingMode = "auto",
+        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
         random_state: int | np.random.RandomState | np.random.Generator | None = 0,
-        n_jobs: Annotated[int | None, deprecated("Use n_preprocessing_jobs")] = None,
-        n_preprocessing_jobs: int = 1,
-        inference_config: dict | InferenceConfig | None = None,
+        n_jobs: int = -1,
+        inference_config: dict | ModelInterfaceConfig | None = None,
         differentiable_input: bool = False,
     ) -> None:
-        """Construct a TabPFN regressor.
-
-        This constructs a regressor using the latest model and settings. If you would
-        like to use a previous model version, use `create_default_for_version()`
-        instead. You can also use `model_path` to specify a particular model.
+        """A TabPFN interface for regression.
 
         Args:
             n_estimators:
@@ -284,18 +250,13 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
                   downloaded to this location.
 
             device:
-                The device(s) to use for inference.
-
-                If "auto": a single device is selected based on availability in the
-                following order of priority: "cuda:0", "mps", "cpu".
-
-                To manually select a single device: specify a PyTorch device string e.g.
-                "cuda:1". See PyTorch's documentation for information about supported
-                devices.
+                The device to use for inference with TabPFN. If set to "auto", the
+                device is selected based on availability in the following order of
+                priority: "cuda", "mps", and then "cpu". You can also set the device
+                manually to one of these options.
 
-                To use several GPUs: specify a list of PyTorch GPU device strings, e.g.
-                ["cuda:0", "cuda:1"]. This can dramatically speed up inference for
-                larger datasets, by executing the estimators in parallel on the GPUs.
+                See PyTorch's documentation on devices for more information about
+                supported devices.
 
             ignore_pretraining_limits:
                 Whether to ignore the pre-training limits of the model. The TabPFN
@@ -312,12 +273,10 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
 
                 !!! note
 
-                    For version 2.5, the pre-training limits are:
+                    The current pre-training limits are:
 
-                    - 50_000 samples/rows
-                    - 2_000 features/columns (Note that for more than 500 features we
-                        subsample 500 features per estimator. It is therefore important
-                        to use a sufficiently large number of `n_estimators`.)
+                    - 10_000 samples/rows
+                    - 500 features/columns
 
             device:
                 The device to use for inference with TabPFN. If `"auto"`, the device is
@@ -371,23 +330,30 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
                   attribute internally.
 
             memory_saving_mode:
-                Enable GPU/CPU memory saving mode. This can both avoid out-of-memory
-                errors and improve fit+predict speed by reducing memory pressure.
-
-                It saves memory by automatically batching certain model computations
-                within TabPFN.
-
-                - If "auto": memory saving mode is enabled/disabled automatically based
-                    on a heuristic
-                - If True/False: memory saving mode is forced enabled/disabled.
-
-                If speed is important to your application, you may wish to manually tune
-                this option by comparing the time taken for fit+predict with it set to
-                False and True.
+                Enable GPU/CPU memory saving mode. This can help to prevent
+                out-of-memory errors that result from computations that would consume
+                more memory than available on the current device. We save memory by
+                automatically batching certain model computations within TabPFN to
+                reduce the total required memory. The options are:
+
+                - If `bool`, enable/disable memory saving mode.
+                - If `"auto"`, we will estimate the amount of memory required for the
+                  forward pass and apply memory saving if it is more than the
+                  available GPU/CPU memory. This is the recommended setting as it
+                  allows for speed-ups and prevents memory errors depending on
+                  the input data.
+                - If `float` or `int`, we treat this value as the maximum amount of
+                  available GPU/CPU memory (in GB). We will estimate the amount
+                  of memory required for the forward pass and apply memory saving
+                  if it is more than this value. Passing a float or int value for
+                  this parameter is the same as setting it to True and explicitly
+                  specifying the maximum free available memory
 
                 !!! warning
                     This does not batch the original input data. We still recommend to
-                    batch the test set as necessary if you run out of memory.
+                    batch this as necessary if you run into memory errors! For example,
+                    if the entire input data does not fit into memory, even the memory
+                    save mode will not prevent memory errors.
 
             random_state:
                 Controls the randomness of the model. Pass an int for reproducible
@@ -408,29 +374,22 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
                     passing `USE_SKLEARN_16_DECIMAL_PRECISION=True` as kwarg.
 
             n_jobs:
-                Deprecated, use `n_preprocessing_jobs` instead.
-                This parameter never had any effect.
-
-            n_preprocessing_jobs:
-                The number of worker processes to use for the preprocessing.
-
-                If `1`, the preprocessing will be performed in the current process,
-                parallelised across multiple CPU cores. If `>1` and `n_estimators > 1`,
-                then different estimators will be dispatched to different processes.
+                The number of workers for tasks that can be parallelized across CPU
+                cores. Currently, this is used for preprocessing the data in parallel
+                (if `n_estimators > 1`).
 
-                We strongly recommend setting this to 1, which has the lowest overhead
-                and can often fully utilise the CPU. Values >1 can help if you have lots
-                of CPU cores available, but can also be slower.
+                - If `-1`, all available CPU cores are used.
+                - If `int`, the number of CPU cores to use is determined by `n_jobs`.
 
             inference_config:
                 For advanced users, additional advanced arguments that adjust the
                 behavior of the model interface.
-                See [tabpfn.inference_config.InferenceConfig][] for details and options.
+                See [tabpfn.constants.ModelInterfaceConfig][] for details and options.
 
-                - If `None`, the default InferenceConfig is used.
+                - If `None`, the default ModelInterfaceConfig is used.
                 - If `dict`, the key-value pairs are used to update the default
-                  `InferenceConfig`. Raises an error if an unknown key is passed.
-                - If `InferenceConfig`, the object is used as the configuration.
+                  `ModelInterfaceConfig`. Raises an error if an unknown key is passed.
+                - If `ModelInterfaceConfig`, the object is used as the configuration.
 
             differentiable_input:
                 If true, preprocessing attempts to be end-to-end differentiable.
@@ -449,120 +408,14 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             inference_precision
         )
         self.fit_mode: Literal["low_memory", "fit_preprocessors", "batched"] = fit_mode
-        self.memory_saving_mode = memory_saving_mode
+        self.memory_saving_mode: bool | Literal["auto"] | float | int = (
+            memory_saving_mode
+        )
         self.random_state = random_state
+        self.n_jobs = n_jobs
         self.inference_config = inference_config
         self.differentiable_input = differentiable_input
 
-        if n_jobs is not None:
-            warnings.warn(
-                "TabPFNRegressor(n_jobs=...) is deprecated and has no effect. "
-                "Use `n_preprocessing_jobs` instead.",
-                DeprecationWarning,
-                stacklevel=2,
-            )
-        self.n_jobs = n_jobs
-        self.n_preprocessing_jobs = n_preprocessing_jobs
-
-        # Ping the usage service if telemetry enabled
-        ping()
-
-    @classmethod
-    def create_default_for_version(cls, version: ModelVersion, **overrides) -> Self:
-        """Construct a regressor that uses the given version of the model.
-
-        In addition to selecting the model, this also configures certain settings to the
-        default values associated with this model version.
-
-        Any kwargs will override the default settings.
-        """
-        if version == ModelVersion.V2:
-            options = {
-                "model_path": str(
-                    get_cache_dir() / ModelSource.get_regressor_v2().default_filename
-                ),
-                "n_estimators": 8,
-                "softmax_temperature": 0.9,
-            }
-        elif version == ModelVersion.V2_5:
-            options = {
-                "model_path": str(
-                    get_cache_dir() / ModelSource.get_regressor_v2_5().default_filename
-                ),
-                "n_estimators": 8,
-                "softmax_temperature": 0.9,
-            }
-        else:
-            raise ValueError(f"Unknown version: {version}")
-
-        options.update(overrides)
-
-        return cls(**options)
-
-    @property
-    def model_(self) -> Architecture:
-        """The model used for inference.
-
-        This is set after the model is loaded and initialized.
-        """
-        if not hasattr(self, "models_"):
-            raise ValueError(
-                "The model has not been initialized yet. Please initialize the model "
-                "before using the `model_` property."
-            )
-        if len(self.models_) > 1:
-            raise ValueError(
-                "The `model_` property is not supported when multiple models are used. "
-                "Use `models_` instead."
-            )
-        return self.models_[0]
-
-    @property
-    def norm_bardist_(self) -> FullSupportBarDistribution:
-        """WARNING: DEPRECATED. Please use `raw_space_bardist_` instead.
-        This attribute will be removed in a future version.
-        """
-        warnings.warn(
-            "`norm_bardist_` is deprecated and will be removed in a future version. "
-            "Please use `raw_space_bardist_` instead.",
-            DeprecationWarning,
-            stacklevel=2,
-        )
-        return self.raw_space_bardist_
-
-    @norm_bardist_.setter
-    def norm_bardist_(self, value: FullSupportBarDistribution) -> None:
-        warnings.warn(
-            "`norm_bardist_` is deprecated and will be removed in a future version. "
-            "Please use `raw_space_bardist_` instead.",
-            DeprecationWarning,
-            stacklevel=2,
-        )
-        self.raw_space_bardist_ = value
-
-    @property
-    def bardist_(self) -> FullSupportBarDistribution:
-        """WARNING: DEPRECATED. Please use `znorm_space_bardist_` instead.
-        This attribute will be removed in a future version.
-        """
-        warnings.warn(
-            "`bardist_` is deprecated and will be removed in a future version. "
-            "Please use `znorm_space_bardist_` instead.",
-            DeprecationWarning,
-            stacklevel=2,
-        )
-        return self.znorm_space_bardist_
-
-    @bardist_.setter
-    def bardist_(self, value: FullSupportBarDistribution) -> None:
-        warnings.warn(
-            "`bardist_` is deprecated and will be removed in a future version. "
-            "Please use `znorm_space_bardist_` instead.",
-            DeprecationWarning,
-            stacklevel=2,
-        )
-        self.znorm_space_bardist_ = value
-
     # TODO: We can remove this from scikit-learn lower bound of 1.6
     def _more_tags(self) -> dict[str, Any]:
         return {
@@ -581,8 +434,6 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         y_raw: YType | list[YType],
         split_fn: Callable,
         max_data_size: None | int = 10000,
-        *,
-        equal_split_size: bool = True,
     ) -> DatasetCollectionWithPreprocessing:
         """Transforms raw input data into a collection of datasets,
         with varying preprocessings.
@@ -601,11 +452,6 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             split_fn: A function to dissect a dataset into train and test partition.
             max_data_size: Maximum allowed number of samples within one dataset.
             If None, datasets are not splitted.
-            equal_split_size: If True, splits data into equally sized chunks under
-            max_data_size.
-            If False, splits into chunks of size `max_data_size`, with
-            the last chunk having the remainder samples but is dropped if its
-            size is less than 2.
         """
         return get_preprocessed_datasets_helper(
             self,
@@ -614,12 +460,11 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             split_fn,
             max_data_size,
             model_type="regressor",
-            equal_split_size=equal_split_size,
         )
 
     def _initialize_model_variables(self) -> tuple[int, np.random.Generator]:
         """Initializes the model, returning byte_size and RNG object."""
-        return initialize_model_variables_helper(self, "regressor")
+        return _initialize_model_variables_helper(self, "regressor")
 
     def _initialize_dataset_preprocessing(
         self, X: XType, y: YType, rng: np.random.Generator
@@ -639,15 +484,15 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             X,
             y,
             estimator=self,
-            ensure_y_numeric=True,
-            max_num_samples=self.inference_config_.MAX_NUMBER_OF_SAMPLES,
-            max_num_features=self.inference_config_.MAX_NUMBER_OF_FEATURES,
+            ensure_y_numeric=False,
+            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
+            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
             ignore_pretraining_limits=self.ignore_pretraining_limits,
         )
 
         assert isinstance(X, np.ndarray)
         check_cpu_warning(
-            self.devices_, X, allow_cpu_override=self.ignore_pretraining_limits
+            self.device, X, allow_cpu_override=self.ignore_pretraining_limits
         )
 
         if feature_names_in is not None:
@@ -657,58 +502,64 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         self.inferred_categorical_indices_ = infer_categorical_features(
             X=X,
             provided=self.categorical_features_indices,
-            min_samples_for_inference=self.inference_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
-            max_unique_for_category=self.inference_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
-            min_unique_for_numerical=self.inference_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
+            min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
+            max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
+            min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
         )
 
         # Will convert inferred categorical indices to category dtype,
         # to be picked up by the ord_encoder, as well
         # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
-        X = fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
+        X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
         # Ensure categories are ordinally encoded
-        ord_encoder = get_ordinal_encoder()
-        X = process_text_na_dataframe(
+        ord_encoder = _get_ordinal_encoder()
+        X = _process_text_na_dataframe(
             X,
             ord_encoder=ord_encoder,
             fit_encoder=True,  # type: ignore
         )
         self.preprocessor_ = ord_encoder
 
-        possible_target_transforms = get_all_reshape_feature_distribution_preprocessors(
-            num_examples=y.shape[0],  # Use length of validated y
-            random_state=rng,  # Use the provided rng
+        possible_target_transforms = (
+            ReshapeFeatureDistributionsStep.get_all_preprocessors(
+                num_examples=y.shape[0],  # Use length of validated y
+                random_state=rng,  # Use the provided rng
+            )
         )
         target_preprocessors: list[TransformerMixin | Pipeline | None] = []
         for (
             y_target_preprocessor
-        ) in self.inference_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS:
+        ) in self.interface_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS:
             if y_target_preprocessor is not None:
                 preprocessor = possible_target_transforms[y_target_preprocessor]
             else:
                 preprocessor = None
             target_preprocessors.append(preprocessor)
+        preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS
 
         ensemble_configs = EnsembleConfig.generate_for_regression(
-            num_estimators=self.n_estimators,
-            subsample_size=self.inference_config_.SUBSAMPLE_SAMPLES,
-            add_fingerprint_feature=self.inference_config_.FINGERPRINT_FEATURE,
-            feature_shift_decoder=self.inference_config_.FEATURE_SHIFT_METHOD,
-            polynomial_features=self.inference_config_.POLYNOMIAL_FEATURES,
+            n=self.n_estimators,
+            subsample_size=self.interface_config_.SUBSAMPLE_SAMPLES,
+            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
+            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
+            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
             max_index=len(X),
-            preprocessor_configs=self.inference_config_.PREPROCESS_TRANSFORMS,
+            preprocessor_configs=typing.cast(
+                "Sequence[PreprocessorConfig]",
+                preprocess_transforms
+                if preprocess_transforms is not None
+                else default_regressor_preprocessor_configs(),
+            ),
             target_transforms=target_preprocessors,
             random_state=rng,
-            num_models=len(self.models_),
         )
 
-        self.znorm_space_bardist_ = self.znorm_space_bardist_.to(self.devices_[0])
+        self.bardist_ = self.bardist_.to(self.device_)
 
         assert len(ensemble_configs) == self.n_estimators
 
-        return ensemble_configs, X, y, self.znorm_space_bardist_
+        return ensemble_configs, X, y, self.bardist_
 
-    @track_model_call("fit", param_names=["X_preprocessed", "y_preprocessed"])
     def fit_from_preprocessed(
         self,
         X_preprocessed: list[torch.Tensor],
@@ -742,11 +593,11 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             self.fit_mode = "batched"
 
         # If there is a model, and we are lazy, we skip reinitialization
-        if not hasattr(self, "models_") or not no_refit:
+        if not hasattr(self, "model_") or not no_refit:
             byte_size, rng = self._initialize_model_variables()
         else:
             _, _, byte_size = determine_precision(
-                self.inference_precision, self.devices_
+                self.inference_precision, self.device_
             )
             rng = None
 
@@ -754,13 +605,13 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         self.executor_ = create_inference_engine(
             X_train=X_preprocessed,
             y_train=y_preprocessed,
-            models=self.models_,
+            model=self.model_,
             ensemble_configs=configs,
             cat_ix=cat_ix,
             fit_mode="batched",
-            devices_=self.devices_,
+            device_=self.device_,
             rng=rng,
-            n_preprocessing_jobs=self.n_preprocessing_jobs,
+            n_jobs=self.n_jobs,
             byte_size=byte_size,
             forced_inference_dtype_=self.forced_inference_dtype_,
             memory_saving_mode=self.memory_saving_mode,
@@ -772,7 +623,6 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         return self
 
     @config_context(transform_output="default")  # type: ignore
-    @track_model_call(model_method="fit", param_names=["X", "y"])
     def fit(self, X: XType, y: YType) -> Self:
         """Fit the model.
 
@@ -793,15 +643,15 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             )
             self.fit_mode = "fit_preprocessors"
 
-        if not hasattr(self, "models_") or not self.differentiable_input:
+        if not hasattr(self, "model_") or not self.differentiable_input:
             byte_size, rng = self._initialize_model_variables()
-            ensemble_configs, X, y, self.znorm_space_bardist_ = (
+            ensemble_configs, X, y, self.bardist_ = (
                 self._initialize_dataset_preprocessing(X, y, rng)
             )
         else:  # already fitted and prompt_tuning mode: no cat. features
             _, rng = infer_random_state(self.random_state)
             _, _, byte_size = determine_precision(
-                self.inference_precision, self.devices_
+                self.inference_precision, self.device_
             )
 
         assert len(ensemble_configs) == self.n_estimators
@@ -810,18 +660,9 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         self.constant_value_ = y[0] if self.is_constant_target_ else None
 
         if self.is_constant_target_:
-            # Use relative epsilon, s.t. it works for small and large constant values
-            border_adjustment = max(
-                abs(self.constant_value_ * REGRESSION_CONSTANT_TARGET_BORDER_EPSILON),
-                REGRESSION_CONSTANT_TARGET_BORDER_EPSILON,
-            )
-
-            self.znorm_space_bardist_ = FullSupportBarDistribution(
+            self.bardist_ = FullSupportBarDistribution(
                 borders=torch.tensor(
-                    [
-                        self.constant_value_ - border_adjustment,
-                        self.constant_value_ + border_adjustment,
-                    ]
+                    [self.constant_value_ - 1e-5, self.constant_value_ + 1e-5]
                 )
             )
             # No need to create an inference engine for a constant prediction
@@ -831,21 +672,21 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         self.y_train_std_ = std.item() + 1e-20
         self.y_train_mean_ = mean.item()
         y = (y - self.y_train_mean_) / self.y_train_std_
-        self.raw_space_bardist_ = FullSupportBarDistribution(
-            self.znorm_space_bardist_.borders * self.y_train_std_ + self.y_train_mean_,
+        self.normalized_bardist_ = FullSupportBarDistribution(
+            self.bardist_.borders * self.y_train_std_ + self.y_train_mean_,
         ).float()
 
         # Create the inference engine
         self.executor_ = create_inference_engine(
             X_train=X,
             y_train=y,
-            models=self.models_,
+            model=self.model_,
             ensemble_configs=ensemble_configs,
             cat_ix=self.inferred_categorical_indices_,
             fit_mode=self.fit_mode,
-            devices_=self.devices_,
+            device_=self.device_,
             rng=rng,
-            n_preprocessing_jobs=self.n_preprocessing_jobs,
+            n_jobs=self.n_jobs,
             byte_size=byte_size,
             forced_inference_dtype_=self.forced_inference_dtype_,
             memory_saving_mode=self.memory_saving_mode,
@@ -892,7 +733,6 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
     ) -> FullOutputDict: ...
 
     @config_context(transform_output="default")  # type: ignore
-    @track_model_call(model_method="predict", param_names=["X"])
     def predict(
         self,
         X: XType,
@@ -940,17 +780,17 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         if quantiles is None:
             quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
         else:
-            assert all((0 <= q <= 1) and (isinstance(q, float)) for q in quantiles), (
-                "All quantiles must be between 0 and 1 and floats."
-            )
+            assert all(
+                (0 <= q <= 1) and (isinstance(q, float)) for q in quantiles
+            ), "All quantiles must be between 0 and 1 and floats."
         if output_type not in _USABLE_OUTPUT_TYPES:
             raise ValueError(f"Invalid output type: {output_type}")
 
         if hasattr(self, "is_constant_target_") and self.is_constant_target_:
             return self._handle_constant_target(X.shape[0], output_type, quantiles)
 
-        X = fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
-        X = process_text_na_dataframe(X, ord_encoder=self.preprocessor_)  # type: ignore
+        X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
+        X = _process_text_na_dataframe(X, ord_encoder=self.preprocessor_)  # type: ignore
 
         # Runs over iteration engine
         (
@@ -963,8 +803,8 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         transformed_logits = [
             translate_probs_across_borders(
                 logits,
-                frm=torch.as_tensor(borders_t, device=logits.device),
-                to=self.znorm_space_bardist_.borders.to(logits.device),
+                frm=torch.as_tensor(borders_t, device=self.device_),
+                to=self.bardist_.borders.to(self.device_),
             )
             for logits, borders_t in zip(outputs, borders)
         ]
@@ -983,7 +823,7 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         logit_to_output = partial(
             _logits_to_output,
             logits=logits,
-            criterion=self.raw_space_bardist_,
+            criterion=self.normalized_bardist_,
             quantiles=quantiles,
         )
         if output_type in ["full", "main"]:
@@ -1011,7 +851,7 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
                 # Return full output with criterion and logits
                 return FullOutputDict(
                     **main_outputs,
-                    criterion=self.raw_space_bardist_,
+                    criterion=self.normalized_bardist_,
                     logits=logits,
                 )
 
@@ -1080,19 +920,18 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
 
         check_is_fitted(self)
 
-        std_borders = self.znorm_space_bardist_.borders.cpu().numpy()
+        std_borders = self.bardist_.borders.cpu().numpy()
         outputs: list[torch.Tensor] = []
         borders: list[np.ndarray] = []
 
         # Iterate over estimators
         for output, config in self.executor_.iter_outputs(
             X,
-            devices=self.devices_,
+            device=self.device_,
             autocast=self.use_autocast_,
         ):
-            output = output.float()  # noqa: PLW2901
             if self.softmax_temperature != 1:
-                output = output / self.softmax_temperature  # noqa: PLW2901
+                output = output.float() / self.softmax_temperature  # noqa: PLW2901
 
             # BSz.= 1 Scenario, the same as normal predict() function
             # Handled by first if-statement
@@ -1121,10 +960,10 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
                     descending_borders = False
                 else:
                     logit_cancel_mask, descending_borders, borders_t = (
-                        transform_borders_one(
+                        _transform_borders_one(
                             std_borders,
                             target_transform=config_for_ensemble.target_transform,
-                            repair_nan_borders_after_transform=self.inference_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
+                            repair_nan_borders_after_transform=self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                         )
                     )
                     if descending_borders:
@@ -1173,7 +1012,7 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
         if output_type == "full":
             return FullOutputDict(
                 **main_outputs,
-                criterion=self.znorm_space_bardist_,
+                criterion=self.bardist_,
                 logits=torch.zeros((n_samples, 1)),
             )
         return main_outputs
@@ -1198,7 +1037,7 @@ class TabPFNRegressor(RegressorMixin, BaseEstimator):
             np.ndarray
                 The computed embeddings for each fitted estimator.
         """
-        return get_embeddings(self, X, data_source)
+        return _get_embeddings(self, X, data_source)
 
     def save_fit_state(self, path: Path | str) -> None:
         """Save a fitted regressor, light wrapper around save_fitted_tabpfn_model."""
diff --git a/src/tabpfn/settings.py b/src/tabpfn/settings.py
index 61f1167..c1b4fb2 100644
--- a/src/tabpfn/settings.py
+++ b/src/tabpfn/settings.py
@@ -3,13 +3,10 @@
 from __future__ import annotations
 
 from pathlib import Path
-from typing import Any
 
-from pydantic import Field, field_validator
+from pydantic import Field
 from pydantic_settings import BaseSettings, SettingsConfigDict
 
-from tabpfn.constants import ModelVersion
-
 
 class TabPFNSettings(BaseSettings):
     """Configuration settings for TabPFN.
@@ -19,11 +16,7 @@ class TabPFNSettings(BaseSettings):
     Prefixed by ``TABPFN_`` in environment variables.
     """
 
-    # Set extra="ignore" so that unknown keys in the .env file, for example, entries for
-    # other applications, do not cause validation errors.
-    model_config = SettingsConfigDict(
-        env_prefix="TABPFN_", env_file=".env", extra="ignore"
-    )
+    model_config = SettingsConfigDict(env_prefix="TABPFN_", env_file=".env")
 
     # Model Configuration
     model_cache_dir: Path | None = Field(
@@ -31,10 +24,6 @@ class TabPFNSettings(BaseSettings):
         description="Custom directory for caching downloaded TabPFN models. "
         "If not set, uses platform-specific user cache directory.",
     )
-    model_version: ModelVersion = Field(
-        default=ModelVersion.V2_5,
-        description="The version of the TabPFN model to use by default.",
-    )
 
     # Performance/Memory Settings
     allow_cpu_large_dataset: bool = Field(
@@ -69,21 +58,6 @@ class TestingSettings(BaseSettings):
         "Typically set by CI systems (e.g., GitHub Actions).",
     )
 
-    @field_validator("ci", mode="before")
-    @classmethod
-    def _parse_ci(cls, value: Any) -> bool:
-        """Interpret any non-empty environment value as ``True``.
-
-        Some CI providers set the ``CI`` environment variable to a non-boolean
-        string (e.g., ``"azure"``).  Treat any non-empty string other than
-        common falsy values as ``True`` so importing TabPFN works seamlessly in
-        those environments.
-        """
-        if isinstance(value, str):
-            value_lower = value.strip().lower()
-            return value_lower not in {"", "0", "false", "no", "off"}
-        return bool(value)
-
 
 class Settings(BaseSettings):
     """Global settings instance."""
diff --git a/src/tabpfn/utils.py b/src/tabpfn/utils.py
index eff69a8..d2ea20f 100644
--- a/src/tabpfn/utils.py
+++ b/src/tabpfn/utils.py
@@ -5,27 +5,28 @@
 from __future__ import annotations
 
 import contextlib
+import ctypes
 import os
 import typing
 from collections.abc import Sequence
-from typing import TYPE_CHECKING, Any, Literal, Union
+from typing import TYPE_CHECKING, Any, Literal
 
 import numpy as np
 import numpy.typing as npt
 import pandas as pd
 import torch
-from sklearn.base import (
-    TransformerMixin,
-    check_is_fitted,
-    is_classifier,
-)
+from sklearn.base import check_is_fitted, is_classifier
+from sklearn.compose import ColumnTransformer, make_column_selector
+from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
 from sklearn.utils.multiclass import check_classification_targets
+from torch import nn
 
 from tabpfn.architectures.base.encoders import (
     MulticlassClassificationTargetEncoder,
     SequentialEncoder,
 )
 from tabpfn.constants import (
+    DEFAULT_NUMPY_PREPROCESSING_DTYPE,
     NA_PLACEHOLDER,
     REGRESSION_NAN_BORDER_LIMIT_LOWER,
     REGRESSION_NAN_BORDER_LIMIT_UPPER,
@@ -34,10 +35,8 @@ from tabpfn.misc._sklearn_compat import check_array, validate_data
 
 if TYPE_CHECKING:
     from sklearn.base import TransformerMixin
-    from sklearn.compose import ColumnTransformer
     from sklearn.pipeline import Pipeline
 
-    from tabpfn.architectures.interface import Architecture
     from tabpfn.classifier import TabPFNClassifier, XType, YType
     from tabpfn.regressor import TabPFNRegressor
 
@@ -61,7 +60,7 @@ def get_autocast_context(
     return torch.autocast(device.type, enabled=enabled)
 
 
-def get_embeddings(
+def _get_embeddings(
     model: TabPFNClassifier | TabPFNRegressor,
     X: XType,
     data_source: Literal["train", "test"] = "test",
@@ -84,7 +83,7 @@ def get_embeddings(
             ``(n_estimators, n_samples, embedding_dim)``. You can average over the
             first axis or reshape to concatenate the estimators, e.g.:
 
-                emb = get_embeddings(model, X)
+                emb = _get_embeddings(model, X)
                 emb_avg = emb.mean(axis=0)
                 emb_concat = emb.reshape(emb.shape[1], -1)
     """
@@ -95,21 +94,19 @@ def get_embeddings(
     selected_data = data_map[data_source]
 
     # Avoid circular imports
-    from tabpfn.preprocessing import (  # noqa: PLC0415
-        ClassifierEnsembleConfig,
-        RegressorEnsembleConfig,
-    )
+    from tabpfn.preprocessing import ClassifierEnsembleConfig, RegressorEnsembleConfig
 
     X = validate_X_predict(X, model)
-    X = fix_dtypes(X, cat_indices=model.categorical_features_indices)
+    X = _fix_dtypes(X, cat_indices=model.categorical_features_indices)
     X = model.preprocessor_.transform(X)
 
     embeddings: list[np.ndarray] = []
 
     # Cast executor to Any to bypass the iter_outputs signature check
-    for output, config in model.executor_.iter_outputs(
+    executor = typing.cast("typing.Any", model.executor_)
+    for output, config in executor.iter_outputs(
         X,
-        devices=model.devices_,
+        device=model.device_,
         autocast=model.use_autocast_,
         only_return_standard_out=False,
     ):
@@ -179,34 +176,31 @@ def _cancel_nan_borders(
     return borders, logit_cancel_mask
 
 
-DevicesSpecification = Union[
-    torch.device, str, Sequence[Union[torch.device, str]], Literal["auto"]
-]
-
-
-def infer_devices(devices: DevicesSpecification) -> tuple[torch.device, ...]:
-    """Selects the appropriate PyTorch devices for inference.
+def infer_device_and_type(device: str | torch.device | None) -> torch.device:
+    """Infers the appropriate PyTorch device based on the input and environment
+    configuration.
 
-    If `device` is "auto" then the devices are selected as follows:
-    1. If CUDA is available and not excluded, returns the first "cuda" device
-    2. Otherwise, if MPS is available and not excluded, returns the "mps" device
-    3. Otherwise, returns the "cpu" device
+    Rules:
+    1. If `device` is `None` or "auto":
+       - Picks "cuda" if available and not excluded via TABPFN_EXCLUDE_DEVICES
+       - Otherwise picks "mps" if available and not excluded
+       - Falls back to "cpu"
+    2. If `device` is a string, converts it to a torch.device
+    3. If already a torch.device, returns as-is
+    4. Otherwise raises ValueError
 
-    CUDA and MPS can be excluded from the "auto" selection by specifying the
-    TABPFN_EXCLUDE_DEVICES environment variable. This can be either "cuda", "mps", or
-    "cuda,mps". This is useful for excluding device classes in CI.
+    Environment:
+        TABPFN_EXCLUDE_DEVICES: comma-separated list of devices to ignore
+        (e.g., "cuda,mps"). This allows excluding "mps" on the CI pipeline.
 
     Args:
-        devices: The device specification. One of:
-            - "auto": the device will be selected as described above
-            - a PyTorch device string like "cuda", "mps", or "cpu": this single device
-                will be selected by parsing the string to a torch.device
-            - a torch.device: this single device will be selected
-            - a list of PyTorch device strings or torch.device: each item will be
-                converted to a torch.device, and all of the devices selected
+        device (str | torch.device | None): The device specification. Can be:
+            - `None` or `"auto"` for automatic inference.
+            - A string like `"cuda"`, `"cpu"`, or `"mps"`.
+            - A `torch.device` instance.
 
     Returns:
-        A tuple of at least one device.
+        The inferred device
     """
     exclude_devices = {
         d.strip()
@@ -214,41 +208,22 @@ def infer_devices(devices: DevicesSpecification) -> tuple[torch.device, ...]:
         if d.strip()
     }
 
-    if devices == "auto":
-        if "cuda" not in exclude_devices and torch.cuda.is_available():
-            return (torch.device("cuda:0"),)
-
-        if torch.backends.mps.is_available() and "mps" not in exclude_devices:
-            return (torch.device("mps"),)
-
-        return (torch.device("cpu"),)
-
-    if isinstance(devices, (str, torch.device)):
-        devices = (devices,)
-
-    devices = tuple(_parse_device(device) for device in devices)
-
-    if len(set(devices)) != len(devices):
-        raise ValueError(
-            "The list of devices for inference cannot contain the same device more "
-            f"than once. It contained: {devices}"
+    if (device is None) or (isinstance(device, str) and device == "auto"):
+        device_type_ = (
+            "cuda"
+            if torch.cuda.is_available() and "cuda" not in exclude_devices
+            else "mps"
+            if torch.backends.mps.is_available() and "mps" not in exclude_devices
+            else "cpu"
         )
+        return torch.device(device_type_)
+    if isinstance(device, str):
+        return torch.device(device)
 
-    return devices
-
+    if isinstance(device, torch.device):
+        return device
 
-def _parse_device(device: str | torch.device) -> torch.device:
-    # This is safe because torch.device(torch.device(...)) is a noop.
-    # torch.device(device) returns a fairly informative error message if `device` is not
-    # a valid device, thus do no extra error reporting.
-    device = torch.device(device)
-
-    # In older versions of PyTorch, some torch.cuda functions fail if the device has no
-    # index. 0 is implicit if no index is specified, so add it.
-    if device == torch.device("cuda"):
-        device = torch.device("cuda:0")
-
-    return device
+    raise ValueError(f"Invalid device: {device}")
 
 
 def is_autocast_available(device_type: str) -> bool:
@@ -284,13 +259,11 @@ def is_autocast_available(device_type: str) -> bool:
         )
 
 
-def infer_fp16_inference_mode(
-    devices: Sequence[torch.device], *, enable: bool | None
-) -> bool:
+def infer_fp16_inference_mode(device: torch.device, *, enable: bool | None) -> bool:
     """Infer whether fp16 inference should be enabled.
 
     Args:
-        devices: The devices to validate against.
+        device: The device to validate against.
         enable:
             Whether it should be enabled, `True` or `False`, otherwise if `None`,
             detect if it's possible and use it if so.
@@ -299,13 +272,12 @@ def infer_fp16_inference_mode(
         Whether to use fp16 inference or not.
 
     Raises:
-        ValueError: If fp16 inference was enabled and any of the selected devices do
-            not support it.
+        ValueError: If fp16 inference was enabled and device type does not support it.
     """
-    is_cpu = any(device.type.lower() == "cpu" for device in devices)
+    is_cpu = device.type.lower() == "cpu"
     fp16_available = (
         not is_cpu  # CPU can show enabled, yet it kills inference speed
-        and any(is_autocast_available(device.type) for device in devices)
+        and is_autocast_available(device.type)
     )
 
     if enable is None:
@@ -316,7 +288,7 @@ def infer_fp16_inference_mode(
             raise ValueError(
                 "You specified `fp16_inference=True`, however"
                 "`torch.amp.autocast_mode.is_autocast_available()`"
-                f" reported that one or more of the selected devices ({devices=})"
+                f" reported that your used device ({device=})"
                 " does not support it."
                 "\nPlease ensure your version of torch and device type"
                 " are compatible with torch.autocast()`"
@@ -337,7 +309,7 @@ STRING_DTYPE_KINDS = "SaU"
 UNSUPPORTED_DTYPE_KINDS = "cM"  # Not needed, just for completeness
 
 
-def fix_dtypes(  # noqa: D103
+def _fix_dtypes(
     X: pd.DataFrame | np.ndarray,
     cat_indices: Sequence[int | str] | None,
     numeric_dtype: Literal["float32", "float64"] = "float64",
@@ -405,6 +377,31 @@ def fix_dtypes(  # noqa: D103
     return X
 
 
+def _get_ordinal_encoder(
+    *,
+    numpy_dtype: np.floating = DEFAULT_NUMPY_PREPROCESSING_DTYPE,  # type: ignore
+) -> ColumnTransformer:
+    oe = OrdinalEncoder(
+        # TODO: Could utilize the categorical dtype values directly instead of "auto"
+        categories="auto",
+        dtype=numpy_dtype,  # type: ignore
+        handle_unknown="use_encoded_value",
+        unknown_value=-1,
+        encoded_missing_value=np.nan,  # Missing stays missing
+    )
+
+    # Documentation of sklearn, deferring to pandas is misleading here. It's done
+    # using a regex on the type of the column, and using `object`, `"object"` and
+    # `np.object` will not pick up strings.
+    to_convert = ["category", "string"]
+    return ColumnTransformer(
+        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
+        remainder=FunctionTransformer(),
+        sparse_threshold=0.0,
+        verbose_feature_names_out=False,
+    )
+
+
 def validate_Xy_fit(
     X: XType,
     y: YType,
@@ -539,23 +536,12 @@ def infer_categorical_features(
     indices = []
 
     for ix, col in enumerate(X.T):
-        # Calculate total distinct values once, treating NaN as a category.
-        try:
-            s = pd.Series(col)
-            # counts NaN/None as a category
-            num_distinct = s.nunique(dropna=False)
-        except TypeError as e:
-            # e.g. "unhashable type: 'dict'" when object arrays contain dicts
-            raise TypeError(
-                "argument must be a string or a number"
-                "(columns must only contain strings or numbers)"
-            ) from e
         if ix in maybe_categoricals:
-            if num_distinct <= max_unique_for_category:
+            if len(np.unique(col)) <= max_unique_for_category:
                 indices.append(ix)
         elif (
             large_enough_x_to_infer_categorical
-            and num_distinct < min_unique_for_numerical
+            and len(np.unique(col)) < min_unique_for_numerical
         ):
             indices.append(ix)
 
@@ -591,22 +577,13 @@ def infer_random_state(
     return static_seed, np_rng
 
 
-def process_text_na_dataframe(
+def _process_text_na_dataframe(  # type: ignore
     X: pd.DataFrame,
     placeholder: str = NA_PLACEHOLDER,
-    ord_encoder: ColumnTransformer | None = None,
+    ord_encoder=None,
     *,
     fit_encoder: bool = False,
 ) -> np.ndarray:
-    """Convert `X` to float64, replacing NA with NaN in string cells.
-
-    If `ord_encoder` is not None, then it will be used to encode `X` before the
-    conversion to float64.
-
-    Note that this function sometimes mutates its input.
-    """
-    # Replace NAN values in X, for dtypes, which the OrdinalEncoder cannot handle
-    # with placeholder NAN value. Later placeholder NAN values are transformed to np.nan
     string_cols = X.select_dtypes(include=["string", "object"]).columns
     if len(string_cols) > 0:
         X[string_cols] = X[string_cols].fillna(placeholder)
@@ -639,7 +616,7 @@ def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
 # However we don't really need the full BarDistribution class and this was
 # put here to make that a bit more obvious in terms of what was going on.
 def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
-    ys = ys.repeat((*logits.shape[:-1], 1))
+    ys = ys.repeat(logits.shape[:-1] + (1,))
     n_bars = len(borders) - 1
     y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)
 
@@ -683,10 +660,11 @@ def translate_probs_across_borders(
 
 
 def update_encoder_params(
-    models: list[Architecture],
+    model: nn.Module,
     remove_outliers_std: float | None,
     seed: int | None,
     *,
+    inplace: Literal[True],
     differentiable_input: bool = False,
 ) -> None:
     """Update the loaded encoder elements and setting to be compatible with inference
@@ -698,58 +676,53 @@ def update_encoder_params(
         This only happens inplace.
 
     Args:
-        models: The models to update.
+        model: The model to update.
         remove_outliers_std: The standard deviation to remove outliers.
         seed: The seed to use, if any.
         inplace: Whether to do the operation inplace.
         differentiable_input: Whether the entire model including forward pass should
             be differentiable with pt autograd. This disables non-differentiable
             encoder steps.
+
+    Raises:
+        ValueError: If `inplace` is not `True`.
     """
+    if not inplace:
+        raise ValueError("Only inplace is supported")
+
     if remove_outliers_std is not None and remove_outliers_std <= 0:
         raise ValueError("remove_outliers_std must be greater than 0")
 
-    for model in models:
-        # TODO: find a less hacky way to change settings during training
-        # and inference
-        if not hasattr(model, "encoder"):
-            raise ValueError(
-                "Model does not have an encoder, this breaks the TabPFN sklearn "
-                "wrapper."
-            )
+    if not hasattr(model, "encoder"):
+        return
 
-        encoder = model.encoder
+    encoder = model.encoder
 
-        # TODO: maybe check that norm_layer even exists
-        norm_layer = next(
-            e for e in encoder if "InputNormalizationEncoderStep" in str(e.__class__)
-        )
-        if not hasattr(norm_layer, "remove_outliers"):
-            raise ValueError(
-                "InputNormalizationEncoderStep does not have a remove_outliers "
-                "attribute, this will break the TabPFN sklearn wrapper."
-            )
-        norm_layer.remove_outliers = (remove_outliers_std is not None) and (
-            remove_outliers_std > 0
-        )
-        if norm_layer.remove_outliers:
-            norm_layer.remove_outliers_sigma = remove_outliers_std
+    # TODO: maybe check that norm_layer even exists
+    norm_layer = next(
+        e for e in encoder if "InputNormalizationEncoderStep" in str(e.__class__)
+    )
+    norm_layer.remove_outliers = (remove_outliers_std is not None) and (
+        remove_outliers_std > 0
+    )
+    if norm_layer.remove_outliers:
+        norm_layer.remove_outliers_sigma = remove_outliers_std
 
-        norm_layer.seed = seed
-        norm_layer.reset_seed()
+    norm_layer.seed = seed
+    norm_layer.reset_seed()
 
-        if differentiable_input:
-            diffable_steps = []  # only differentiable encoder steps.
-            for module in model.y_encoder:
-                if isinstance(module, MulticlassClassificationTargetEncoder):
-                    pass
-                else:
-                    diffable_steps.append(module)
+    if differentiable_input:
+        diffable_steps = []  # only differentiable encoder steps.
+        for module in model.y_encoder:
+            if isinstance(module, MulticlassClassificationTargetEncoder):
+                pass
+            else:
+                diffable_steps.append(module)
 
-            model.y_encoder = SequentialEncoder(*diffable_steps)
+        model.y_encoder = SequentialEncoder(*diffable_steps)
 
 
-def transform_borders_one(
+def _transform_borders_one(
     borders: np.ndarray,
     target_transform: TransformerMixin | Pipeline,
     *,
@@ -798,13 +771,51 @@ def transform_borders_one(
     return logit_cancel_mask, descending_borders, borders_t
 
 
-def split_large_data(
-    largeX: XType,
-    largey: YType,
-    max_data_size: int,
-    *,
-    equal_split_size: bool,
-) -> tuple[list[XType], list[YType]]:
+# Terminology: Use memory to referent physical memory, swap for swap memory
+def get_total_memory_windows() -> float:
+    """Get the total memory of the system for windows OS, using windows API.
+
+    Returns:
+        The total memory of the system in GB.
+    """
+    import platform
+
+    if platform.system() != "Windows":
+        return 0.0  # Function should not be called on non-Windows platforms
+
+    # ref: https://github.com/microsoft/windows-rs/blob/c9177f7a65c764c237a9aebbd3803de683bedaab/crates/tests/bindgen/src/fn_return_void_sys.rs#L12
+    # ref: https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
+    # this class is needed to load the memory status with GlobalMemoryStatusEx function
+    # using win32 API, for more details see microsoft docs link above
+    class _MEMORYSTATUSEX(ctypes.Structure):
+        _fields_: typing.ClassVar = [
+            ("dwLength", ctypes.c_ulong),
+            ("dwMemoryLoad", ctypes.c_ulong),
+            ("ullTotalPhys", ctypes.c_ulonglong),
+            ("ullAvailPhys", ctypes.c_ulonglong),
+            ("ullTotalPageFile", ctypes.c_ulonglong),
+            ("ullAvailPageFile", ctypes.c_ulonglong),
+            ("ullTotalVirtual", ctypes.c_ulonglong),
+            ("ullAvailVirtual", ctypes.c_ulonglong),
+            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
+        ]
+
+    # Initialize the structure
+    mem_status = _MEMORYSTATUSEX()
+    # need to initialize length of structure, see Microsoft docs above
+    mem_status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
+    try:
+        # Use typing.cast to help mypy understand this Windows-only code
+        windll = typing.cast("typing.Any", ctypes).windll
+        k32_lib = windll.LoadLibrary("kernel32.dll")
+        k32_lib.GlobalMemoryStatusEx(ctypes.byref(mem_status))
+        return float(mem_status.ullTotalPhys) / 1e9  # Convert bytes to GB
+    except (AttributeError, OSError):
+        # Fall back if not on Windows or if the function fails
+        return 0.0
+
+
+def split_large_data(largeX: XType, largey: YType, max_data_size: int):
     """Split a large dataset into chunks along the first dimension.
 
     Args:
@@ -813,33 +824,12 @@ def split_large_data(
         max_data_size: int that indicates max size of a chunks.
             We chose the minimum number of chunks that keeps each chunk under
             max_data_size.
-        equal_split_size: If True, splits data into equally sized chunks under
-            max_data_size.
-            If False, splits into chunks of size `max_data_size`, with
-            the last chunk having the remainder samples but is dropped if its
-            size is less than 2.
     """
     tot_size = len(largeX)
     if max_data_size <= 0:
         raise ValueError("max_data_size must be positive")
     if tot_size == 0:
         return [], []
-
-    if not equal_split_size:
-        MIN_BATCH_SIZE = 2
-
-        xlst, ylst = [], []
-        offset = 0
-        while offset + max_data_size <= tot_size:
-            xlst.append(largeX[offset : offset + max_data_size])
-            ylst.append(largey[offset : offset + max_data_size])
-            offset += max_data_size
-
-        if tot_size - offset >= MIN_BATCH_SIZE:
-            xlst.append(largeX[offset:])
-            ylst.append(largey[offset:])
-
-        return xlst, ylst
     num_chunks = ((tot_size - 1) // max_data_size) + 1
     basechunk_size = tot_size // num_chunks
     remainder = tot_size % num_chunks
@@ -854,12 +844,7 @@ def split_large_data(
     return xlst, ylst
 
 
-def pad_tensors(
-    tensor_list: list[torch.Tensor],
-    padding_val: float | None = 0,
-    *,
-    labels: bool = False,
-) -> list[torch.Tensor]:
+def pad_tensors(tensor_list, padding_val=0, *, labels=False):
     """Pad tensors to maximum dims at the last dimensions.
     if labels=False, 2d tensors are expected, if labels=True, one 1d
     vectors are expected as inputs.
@@ -886,7 +871,7 @@ def pad_tensors(
     return ret_list
 
 
-def meta_dataset_collator(batch: list, padding_val: float = 0.0) -> tuple:
+def meta_dataset_collator(batch, padding_val=0.0):
     """Collate function for torch.utils.data.DataLoader.
 
     Designed for batches from DatasetCollectionWithPreprocessing.
@@ -954,23 +939,3 @@ def meta_dataset_collator(batch: list, padding_val: float = 0.0) -> tuple:
             items_list.append([batch[r][item_idx] for r in range(batch_sz)])
 
     return tuple(items_list)
-
-
-def balance_probas_by_class_counts(
-    probas: torch.Tensor,
-    class_counts: np.ndarray,
-) -> torch.Tensor:
-    """Balance probabilities by class counts.
-
-    Args:
-        probas: The probabilities to balance.
-        class_counts: The class counts to use for balancing.
-
-    Returns:
-        The balanced probabilities.
-    """
-    class_prob_in_train = class_counts / class_counts.sum()
-    balanced_probas = probas / torch.from_numpy(class_prob_in_train).float().to(
-        probas.device
-    )
-    return balanced_probas / balanced_probas.sum(dim=-1, keepdim=True)
diff --git a/tests/conftest.py b/tests/conftest.py
deleted file mode 100644
index b4b5f98..0000000
--- a/tests/conftest.py
+++ /dev/null
@@ -1,29 +0,0 @@
-"""Pytest configuration for TabPFN tests.
-
-This module sets up global test configuration, including disabling telemetry
-for all tests to ensure consistent behavior and avoid external dependencies
-during testing.
-"""
-
-from __future__ import annotations
-
-import os
-import random
-
-import numpy as np
-import pytest
-import torch
-
-
-def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
-    """Configure pytest with global settings."""
-    # Disable telemetry for all tests to ensure consistent behavior
-    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
-
-
-@pytest.fixture(autouse=True, scope="function")  # noqa: PT003
-def set_global_seed() -> None:
-    seed = 42
-    torch.manual_seed(seed)
-    np.random.seed(seed)  # noqa: NPY002
-    random.seed(seed)
diff --git a/tests/reference_predictions/ensemble_classifier_predictions.json b/tests/reference_predictions/ensemble_classifier_predictions.json
index 303cc48..b10b56c 100644
--- a/tests/reference_predictions/ensemble_classifier_predictions.json
+++ b/tests/reference_predictions/ensemble_classifier_predictions.json
@@ -1,14 +1,14 @@
 [
   [
-    0.6467362642288208,
-    0.353263795375824
+    0.6512526273727417,
+    0.3487473428249359
   ],
   [
-    0.6103242635726929,
-    0.3896757662296295
+    0.621411144733429,
+    0.3785889148712158
   ],
   [
-    0.5759167671203613,
-    0.42408323287963867
+    0.5852805972099304,
+    0.4147194027900696
   ]
 ]
\ No newline at end of file
diff --git a/tests/reference_predictions/inconsistency_test_predictions.json b/tests/reference_predictions/inconsistency_test_predictions.json
index 6347392..93d588d 100644
--- a/tests/reference_predictions/inconsistency_test_predictions.json
+++ b/tests/reference_predictions/inconsistency_test_predictions.json
@@ -1,14 +1,14 @@
 [
   [
-    0.6948885917663574,
-    0.30511146783828735
+    0.6066580414772034,
+    0.39334195852279663
   ],
   [
-    0.6349897384643555,
-    0.36501023173332214
+    0.5872554183006287,
+    0.4127446115016937
   ],
   [
-    0.5493558645248413,
-    0.45064419507980347
+    0.515444815158844,
+    0.4845552146434784
   ]
 ]
\ No newline at end of file
diff --git a/tests/reference_predictions/iris_multiclass_predictions.json b/tests/reference_predictions/iris_multiclass_predictions.json
index 03dfd87..2b898ba 100644
--- a/tests/reference_predictions/iris_multiclass_predictions.json
+++ b/tests/reference_predictions/iris_multiclass_predictions.json
@@ -1,17 +1,17 @@
 [
   [
-    0.9999802112579346,
-    1.57855265570106e-05,
-    3.991436642536428e-06
+    0.999957799911499,
+    1.9721099306480028e-05,
+    2.2478077880805358e-05
   ],
   [
-    0.00032155567896552384,
-    0.9981730580329895,
-    0.001505352440290153
+    0.000542996684089303,
+    0.9991786479949951,
+    0.00027837802190333605
   ],
   [
-    0.00041552429320290685,
-    0.004381614737212658,
-    0.995202898979187
+    0.0012329068267717957,
+    0.0033026989549398422,
+    0.9954643845558167
   ]
 ]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_classifier_fit_preprocessors_predictions.json b/tests/reference_predictions/tiny_classifier_fit_preprocessors_predictions.json
deleted file mode 100644
index 6347392..0000000
--- a/tests/reference_predictions/tiny_classifier_fit_preprocessors_predictions.json
+++ /dev/null
@@ -1,14 +0,0 @@
-[
-  [
-    0.6948885917663574,
-    0.30511146783828735
-  ],
-  [
-    0.6349897384643555,
-    0.36501023173332214
-  ],
-  [
-    0.5493558645248413,
-    0.45064419507980347
-  ]
-]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_classifier_fit_with_cache_predictions.json b/tests/reference_predictions/tiny_classifier_fit_with_cache_predictions.json
deleted file mode 100644
index a6f4ea5..0000000
--- a/tests/reference_predictions/tiny_classifier_fit_with_cache_predictions.json
+++ /dev/null
@@ -1,14 +0,0 @@
-[
-  [
-    0.5087710022926331,
-    0.49122902750968933
-  ],
-  [
-    0.5034657120704651,
-    0.4965342879295349
-  ],
-  [
-    0.48383021354675293,
-    0.5161697864532471
-  ]
-]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_classifier_low_memory_predictions.json b/tests/reference_predictions/tiny_classifier_low_memory_predictions.json
deleted file mode 100644
index b2539e7..0000000
--- a/tests/reference_predictions/tiny_classifier_low_memory_predictions.json
+++ /dev/null
@@ -1,14 +0,0 @@
-[
-  [
-    0.5994216203689575,
-    0.4005783498287201
-  ],
-  [
-    0.5221465826034546,
-    0.4778534173965454
-  ],
-  [
-    0.5693687796592712,
-    0.43063122034072876
-  ]
-]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_classifier_predictions.json b/tests/reference_predictions/tiny_classifier_predictions.json
new file mode 100644
index 0000000..93d588d
--- /dev/null
+++ b/tests/reference_predictions/tiny_classifier_predictions.json
@@ -0,0 +1,14 @@
+[
+  [
+    0.6066580414772034,
+    0.39334195852279663
+  ],
+  [
+    0.5872554183006287,
+    0.4127446115016937
+  ],
+  [
+    0.515444815158844,
+    0.4845552146434784
+  ]
+]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_diff_input_classifier_predictions.json b/tests/reference_predictions/tiny_diff_input_classifier_predictions.json
index 6fb1a1c..fe7fb48 100644
--- a/tests/reference_predictions/tiny_diff_input_classifier_predictions.json
+++ b/tests/reference_predictions/tiny_diff_input_classifier_predictions.json
@@ -1,14 +1,14 @@
 [
   [
-    0.5189114809036255,
-    0.4810885787010193
+    0.549764096736908,
+    0.45023590326309204
   ],
   [
-    0.6187573075294495,
-    0.3812427222728729
+    0.6223490238189697,
+    0.3776509761810303
   ],
   [
-    0.42232125997543335,
-    0.5776787400245667
+    0.4886346459388733,
+    0.5113654136657715
   ]
 ]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_regressor_fit_preprocessors_predictions.json b/tests/reference_predictions/tiny_regressor_fit_preprocessors_predictions.json
deleted file mode 100644
index a1e83bc..0000000
--- a/tests/reference_predictions/tiny_regressor_fit_preprocessors_predictions.json
+++ /dev/null
@@ -1,5 +0,0 @@
-[
-  4.442662715911865,
-  5.8899030685424805,
-  8.630510330200195
-]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_regressor_fit_with_cache_predictions.json b/tests/reference_predictions/tiny_regressor_fit_with_cache_predictions.json
deleted file mode 100644
index d30b903..0000000
--- a/tests/reference_predictions/tiny_regressor_fit_with_cache_predictions.json
+++ /dev/null
@@ -1,5 +0,0 @@
-[
-  5.399849891662598,
-  5.208672046661377,
-  5.431334972381592
-]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_regressor_low_memory_predictions.json b/tests/reference_predictions/tiny_regressor_low_memory_predictions.json
deleted file mode 100644
index 902de48..0000000
--- a/tests/reference_predictions/tiny_regressor_low_memory_predictions.json
+++ /dev/null
@@ -1,5 +0,0 @@
-[
-  4.36435604095459,
-  6.005045413970947,
-  8.588580131530762
-]
\ No newline at end of file
diff --git a/tests/reference_predictions/tiny_regressor_predictions.json b/tests/reference_predictions/tiny_regressor_predictions.json
new file mode 100644
index 0000000..f031bfb
--- /dev/null
+++ b/tests/reference_predictions/tiny_regressor_predictions.json
@@ -0,0 +1,5 @@
+[
+  5.580631,
+  6.600795,
+  8.451426
+]
diff --git a/tests/test_classifier_interface.py b/tests/test_classifier_interface.py
index 44acb04..1c4a48b 100644
--- a/tests/test_classifier_interface.py
+++ b/tests/test_classifier_interface.py
@@ -2,6 +2,7 @@ from __future__ import annotations
 
 import io
 import os
+import sys
 import typing
 from itertools import product
 from typing import Callable, Literal
@@ -20,33 +21,26 @@ from sklearn.utils.estimator_checks import parametrize_with_checks
 from torch import nn
 
 from tabpfn import TabPFNClassifier
-from tabpfn.architectures import base
-from tabpfn.architectures.base.config import ModelConfig
 from tabpfn.base import ClassifierModelSpecs, initialize_tabpfn_model
-from tabpfn.constants import ModelVersion
-from tabpfn.inference_config import InferenceConfig
-from tabpfn.inference_tuning import (
-    MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING,
-    ClassifierEvalMetrics,
-    ClassifierTuningConfig,
-)
-from tabpfn.model_loading import ModelSource
 from tabpfn.preprocessing import PreprocessorConfig
-from tabpfn.utils import infer_devices
+from tabpfn.utils import infer_device_and_type
 
-from .utils import check_cpu_float16_support, get_pytest_devices
+from .utils import check_cpu_float16_support
 
 exclude_devices = {
     d.strip() for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",") if d.strip()
 }
 
-devices = get_pytest_devices()
+devices = ["cpu"]
+if torch.cuda.is_available() and "cuda" not in exclude_devices:
+    devices.append("cuda")
+if torch.backends.mps.is_available() and "mps" not in exclude_devices:
+    devices.append("mps")
 
 is_cpu_float16_supported = check_cpu_float16_support()
 
-# --- Define parameter combinations ---
-# These are the parameters we want to test in our grid search
 # TODO: test "batched" mode
+
 feature_shift_decoders = ["shuffle", "rotate"]
 multiclass_decoders = ["shuffle", "rotate"]
 fit_modes = [
@@ -58,78 +52,33 @@ inference_precision_methods = ["auto", "autocast", torch.float64, torch.float16]
 remove_outliers_stds = [None, 12]
 estimators = [1, 2]
 
-model_paths = ModelSource.get_classifier_v2().filenames
-primary_model = ModelSource.get_classifier_v2().default_filename
-other_models = [model_path for model_path in model_paths if model_path != primary_model]
-
-# --- Build parameter combinations ---
-# Full grid for the first (primary) model path
-_full_grid = product(
-    estimators,
-    devices,  # device
-    feature_shift_decoders,
-    multiclass_decoders,
-    fit_modes,
-    inference_precision_methods,
-    remove_outliers_stds,
-    [primary_model],  # only the first entry
-)
-
-# Minimal "smoke" grid for all remaining model paths (one combo per path)
-_smoke_grid = product(
-    [1],  # n_estimators
-    ["cpu"],  # device (fast & universally available)
-    ["shuffle"],  # feature_shift_decoder
-    ["shuffle"],  # multiclass_decoder
-    ["fit_preprocessors"],  # fit_mode
-    ["auto"],  # inference_precision
-    [None],  # remove_outliers_std
-    # every non-first model path and multiple models test
-    [*other_models, [primary_model, other_models[0]]],
+all_combinations = list(
+    product(
+        estimators,
+        devices,
+        feature_shift_decoders,
+        multiclass_decoders,
+        fit_modes,
+        inference_precision_methods,
+        remove_outliers_stds,
+    ),
 )
 
-all_combinations = list(_full_grid) + list(_smoke_grid)
-
 
 @pytest.fixture(scope="module")
 def X_y() -> tuple[np.ndarray, np.ndarray]:
-    n_classes = 3
-    return sklearn.datasets.make_classification(
-        n_samples=20 * n_classes,
-        n_classes=n_classes,
-        n_features=5,
-        n_informative=5,
-        n_redundant=0,
-        random_state=0,
-    )
+    X, y = sklearn.datasets.load_iris(return_X_y=True)
+    # Take 20 samples from class 0, 20 from class 1, 20 from class 2
+    # This ensures all 3 classes are present
+    X_diverse = np.vstack([X[y == 0][:20], X[y == 1][:20], X[y == 2][:20]])
+    y_diverse = np.hstack([y[y == 0][:20], y[y == 1][:20], y[y == 2][:20]])
 
+    # Shuffle to mix them up, otherwise training data would be ordered by class
+    indices = np.arange(len(y_diverse))
+    rng = np.random.default_rng(42)
+    rng.shuffle(indices)
 
-def _create_dummy_classifier_model_specs(
-    max_num_classes: int = 10,
-) -> ClassifierModelSpecs:
-    minimal_config = ModelConfig(
-        emsize=8,
-        features_per_group=1,
-        max_num_classes=max_num_classes,
-        nhead=2,
-        nlayers=2,
-        remove_duplicate_features=True,
-        num_buckets=100,
-    )
-    model = base.get_architecture(
-        config=minimal_config,
-        n_out=max_num_classes,
-        cache_trainset_representation=False,
-    )
-    inference_config = InferenceConfig.get_default(
-        task_type="multiclass",
-        model_version=ModelVersion.V2_5,
-    )
-    return ClassifierModelSpecs(
-        model=model,
-        architecture_config=minimal_config,
-        inference_config=inference_config,
-    )
+    return X_diverse[indices].astype(np.float32), y_diverse[indices].astype(np.int64)
 
 
 @pytest.mark.parametrize(
@@ -141,7 +90,6 @@ def _create_dummy_classifier_model_specs(
         "fit_mode",
         "inference_precision",
         "remove_outliers_std",
-        "model_path",
     ),
     all_combinations,
 )
@@ -153,27 +101,22 @@ def test_fit(
     fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
     inference_precision: torch.types._dtype | Literal["autocast", "auto"],
     remove_outliers_std: int | None,
-    model_path: str,
     X_y: tuple[np.ndarray, np.ndarray],
 ) -> None:
-    if inference_precision == "autocast":
-        if torch.device(device).type == "cpu":
-            pytest.skip("CPU device does not support 'autocast' inference.")
-        if torch.device(device).type == "mps" and torch.__version__ < "2.5":
-            pytest.skip("MPS does not support mixed precision before PyTorch 2.5")
+    if device == "cpu" and inference_precision in ["autocast"]:
+        pytest.skip("CPU device does not support 'autocast' inference.")
 
     # Use the environment-aware check to skip only if necessary
     if (
-        torch.device(device).type == "cpu"
+        device == "cpu"
         and inference_precision == torch.float16
         and not is_cpu_float16_supported
     ):
         pytest.skip("CPU float16 matmul not supported in this PyTorch version.")
-    if torch.device(device).type == "mps" and inference_precision == torch.float64:
+    if device == "mps" and inference_precision == torch.float64:
         pytest.skip("MPS does not support float64, which is required for this check.")
 
     model = TabPFNClassifier(
-        model_path=model_path,
         n_estimators=n_estimators,
         device=device,
         fit_mode=fit_mode,
@@ -212,13 +155,14 @@ def test_fit(
     list(
         product(
             [1, 4],  # n_estimators
-            devices,  # device
+            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],  # device
             [0.5, 1.0, 1.5],  # softmax_temperature
             [False, True],  # average_before_softmax
         )
     ),
 )
 def test_predict_logits_and_consistency(
+    X_y: tuple[np.ndarray, np.ndarray],
     n_estimators,
     device,
     softmax_temperature,
@@ -228,15 +172,7 @@ def test_predict_logits_and_consistency(
     under various configuration permutations that affect the post-processing
     pipeline.
     """
-    X, y = sklearn.datasets.make_classification(
-        n_samples=80,
-        n_classes=3,
-        n_features=3,
-        n_informative=3,
-        n_redundant=0,
-        n_clusters_per_class=1,
-        random_state=42,
-    )
+    X, y = X_y
 
     # Ensure y is int64 for consistency with classification tasks
     y = y.astype(np.int64)
@@ -246,7 +182,10 @@ def test_predict_logits_and_consistency(
         device=device,
         softmax_temperature=softmax_temperature,
         average_before_softmax=average_before_softmax,
-        random_state=42,
+        # Disable SKLEARN_16_DECIMAL_PRECISION for this test to avoid rounding
+        # differences in predict_proba's internal output for comparison
+        inference_config={"USE_SKLEARN_16_DECIMAL_PRECISION": False},
+        random_state=42,  # Ensure reproducibility
     )
     classifier.fit(X, y)
 
@@ -278,8 +217,8 @@ def test_predict_logits_and_consistency(
         np.testing.assert_allclose(
             proba_from_logits,
             proba_from_predict_proba,
-            atol=0.0001,
-            rtol=0.0005,
+            atol=1e-5,
+            rtol=1e-5,
             err_msg=(
                 "Probabilities derived from predict_logits do not match "
                 "predict_proba output when they should be consistent."
@@ -290,7 +229,15 @@ def test_predict_logits_and_consistency(
         # softmax to the averaged logits will NOT match predict_proba.
         # predict_proba averages the probabilities, not the logits.
         # softmax(avg(logits)) != avg(softmax(logits))
-        pass
+        proba_from_logits = torch.nn.functional.softmax(
+            torch.from_numpy(logits), dim=-1
+        ).numpy()
+        assert not np.allclose(
+            proba_from_logits, proba_from_predict_proba, atol=1e-5, rtol=1e-5
+        ), (
+            "Outputs unexpectedly matched when averaging after softmax, "
+            "indicating the logic path might be incorrect."
+        )
 
     # 3. Quick check of predict  for completeness, derived from predict_proba
     predicted_labels = classifier.predict(X)
@@ -305,70 +252,6 @@ def test_predict_logits_and_consistency(
     assert log_loss(y, proba_from_predict_proba) < 5.0
 
 
-@pytest.mark.parametrize(("n_estimators"), [1, 2])
-def test_predict_raw_logits(
-    X_y: tuple[np.ndarray, np.ndarray],
-    n_estimators: int,
-):
-    """Tests the predict_raw_logits method."""
-    X, y = X_y
-
-    # Ensure y is int64 for consistency with classification tasks
-    y = y.astype(np.int64)
-
-    classifier = TabPFNClassifier(
-        n_estimators=n_estimators,
-        random_state=42,
-    )
-    classifier.fit(X, y)
-
-    logits = classifier.predict_raw_logits(X)
-    assert logits.shape[0] == n_estimators
-    assert isinstance(logits, np.ndarray)
-    assert logits.shape == (n_estimators, X.shape[0], classifier.n_classes_)
-    assert logits.dtype == np.float32
-    assert not np.isnan(logits).any()
-    assert not np.isinf(logits).any()
-    if classifier.n_classes_ > 1:
-        assert not np.all(logits == logits[:, 0:1]), (
-            "Logits are identical across classes for all samples, indicating "
-            "trivial output."
-        )
-
-
-def test_multiple_models_predict_different_logits(X_y: tuple[np.ndarray, np.ndarray]):
-    """Tests the predict_raw_logits method."""
-    X, y = X_y
-
-    single_model = primary_model
-    two_identical_models = [primary_model, primary_model]
-    two_different_models = [primary_model, other_models[0]]
-
-    # Ensure y is int64 for consistency with classification tasks
-    y = y.astype(np.int64)
-
-    def get_averaged_logits(model_paths: list[str]) -> np.ndarray:
-        classifier = TabPFNClassifier(
-            n_estimators=2,
-            random_state=42,
-            model_path=model_paths,
-        )
-        classifier.fit(X, y)
-        # shape: E=estimators, R=rows, C=columns
-        logits_ERC = classifier.predict_raw_logits(X)
-        return logits_ERC.mean(axis=0)
-
-    single_model_logits = get_averaged_logits(model_paths=[single_model])
-    two_identical_models_logits = get_averaged_logits(model_paths=two_identical_models)
-    two_different_models_logits = get_averaged_logits(model_paths=two_different_models)
-
-    assert not np.all(single_model_logits == single_model_logits[:, 0:1]), (
-        "Logits are identical across classes for all samples, indicating trivial output"
-    )
-    assert np.all(single_model_logits == two_identical_models_logits)
-    assert not np.all(single_model_logits == two_different_models_logits)
-
-
 def test_softmax_temperature_impact_on_logits_magnitude(
     X_y: tuple[np.ndarray, np.ndarray],
 ):
@@ -392,9 +275,9 @@ def test_softmax_temperature_impact_on_logits_magnitude(
     model_high_temp.fit(X, y)
     logits_high_temp = model_high_temp.predict_logits(X)
 
-    assert np.mean(np.abs(logits_low_temp)) > np.mean(np.abs(logits_high_temp)), (
-        "Low softmax temperature did not result in more extreme logits."
-    )
+    assert np.mean(np.abs(logits_low_temp)) > np.mean(
+        np.abs(logits_high_temp)
+    ), "Low softmax temperature did not result in more extreme logits."
 
     model_temp_one = TabPFNClassifier(
         softmax_temperature=1.0, n_estimators=1, device="cpu", random_state=42
@@ -402,12 +285,12 @@ def test_softmax_temperature_impact_on_logits_magnitude(
     model_temp_one.fit(X, y)
     logits_temp_one = model_temp_one.predict_logits(X)
 
-    assert not np.allclose(logits_temp_one, logits_low_temp, atol=1e-6), (
-        "Logits did not change with low temperature."
-    )
-    assert not np.allclose(logits_temp_one, logits_high_temp, atol=1e-6), (
-        "Logits did not change with high temperature."
-    )
+    assert not np.allclose(
+        logits_temp_one, logits_low_temp, atol=1e-6
+    ), "Logits did not change with low temperature."
+    assert not np.allclose(
+        logits_temp_one, logits_high_temp, atol=1e-6
+    ), "Logits did not change with high temperature."
 
 
 def test_balance_probabilities_alters_proba_output(
@@ -416,7 +299,7 @@ def test_balance_probabilities_alters_proba_output(
     """Verifies that enabling `balance_probabilities` indeed changes the output
     probabilities (assuming non-uniform class counts).
     """
-    X_full, _y_full = X_y
+    X_full, y_full = X_y
 
     # Introduce artificial imbalance to ensure balancing has an effect
     y_imbalanced = np.array(
@@ -445,18 +328,16 @@ def test_balance_probabilities_alters_proba_output(
     model_balance.fit(X_subset, y_imbalanced)
     proba_balance = model_balance.predict_proba(X_subset)
 
-    assert not np.allclose(proba_no_balance, proba_balance, atol=1e-5), (
-        "Probabilities did not change when balance_probabilities was toggled."
-    )
+    assert not np.allclose(
+        proba_no_balance, proba_balance, atol=1e-5
+    ), "Probabilities did not change when balance_probabilities was toggled."
 
 
-@pytest.mark.skip(
-    reason="The result is actually different depending on the fitting mode."
-)
+@pytest.mark.skip(reason="No longer passes now dataset has been shrunk.")
 def test_fit_modes_all_return_equal_results(
     X_y: tuple[np.ndarray, np.ndarray],
 ) -> None:
-    kwargs = {"n_estimators": 2, "device": "auto", "random_state": 0}
+    kwargs = {"n_estimators": 2, "device": "cpu", "random_state": 0}
     X, y = X_y
 
     torch.random.manual_seed(0)
@@ -491,19 +372,10 @@ def test_sklearn_compatible_estimator(
     estimator: TabPFNClassifier,
     check: Callable[[TabPFNClassifier], None],
 ) -> None:
-    _auto_devices = infer_devices(devices="auto")
-    if any(device.type == "mps" for device in _auto_devices):
+    _auto_device = infer_device_and_type(device="auto")
+    if _auto_device.type == "mps":
         pytest.skip("MPS does not support float64, which is required for this check.")
 
-    if (
-        check.func.__name__ == "check_classifiers_train"  # type: ignore
-        and _auto_devices[0].type == "cpu"
-    ):
-        pytest.skip(
-            "We currently skip this check on CPU because CPU inference with "
-            "float64 is brokwn for datasets with small number of features."
-        )
-
     if check.func.__name__ in (  # type: ignore
         "check_methods_subset_invariance",
         "check_methods_sample_order_invariance",
@@ -513,55 +385,32 @@ def test_sklearn_compatible_estimator(
     check(estimator)
 
 
-@pytest.mark.skip(reason="This test is flaky and needs to be fixed.")
-def test_balanced_probabilities() -> None:
+def test_balanced_probabilities(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     """Test that balance_probabilities=True works correctly."""
-    n_classes = 3
-    n_features = 3
-
-    # Create an IMBALANCED dataset
-    X, y = sklearn.datasets.make_classification(
-        n_samples=60,
-        n_classes=n_classes,
-        n_features=n_features,
-        n_informative=n_features,
-        n_redundant=0,
-        weights=[0.8, 0.1, 0.1],
-        random_state=42,
-    )
-
-    model_unbalanced = TabPFNClassifier(
-        balance_probabilities=False,
-        random_state=42,
-        n_estimators=2,
-    )
-    model_unbalanced.fit(X, y)
-    proba_unbalanced = model_unbalanced.predict_proba(X)
+    X, y = X_y
 
-    model_balanced = TabPFNClassifier(
+    model = TabPFNClassifier(
         balance_probabilities=True,
-        random_state=42,
-        n_estimators=2,
     )
-    model_balanced.fit(X, y)
-    proba_balanced = model_balanced.predict_proba(X)
 
-    mean_proba_unbalanced = proba_unbalanced.mean(axis=0)
-    mean_proba_balanced = proba_balanced.mean(axis=0)
+    model.fit(X, y)
+    probabilities = model.predict_proba(X)
 
-    # Balanced should be MORE uniform than unbalanced
-    balanced_deviation = np.std(mean_proba_balanced)
-    unbalanced_deviation = np.std(mean_proba_unbalanced)
-    assert balanced_deviation < unbalanced_deviation, (
-        "Balancing did not make probabilities more uniform"
-    )
+    assert np.allclose(probabilities.sum(axis=1), 1.0)
+
+    mean_probs = probabilities.mean(axis=0)
+    expected_mean = 1.0 / len(np.unique(y))
+    assert np.allclose(
+        mean_probs,
+        expected_mean,
+        rtol=0.1,
+    ), "Class probabilities are not properly balanced"
 
 
 def test_classifier_in_pipeline(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     """Test that TabPFNClassifier works correctly within a sklearn pipeline."""
     X, y = X_y
 
-    # Create a simple preprocessing pipeline
     pipeline = Pipeline(
         [
             ("scaler", StandardScaler()),
@@ -577,22 +426,27 @@ def test_classifier_in_pipeline(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     pipeline.fit(X, y)
     probabilities = pipeline.predict_proba(X)
 
-    # Check that probabilities sum to 1 for each prediction
     assert np.allclose(probabilities.sum(axis=1), 1.0)
-    assert probabilities.shape == (X.shape[0], len(np.unique(y)))
+
+    mean_probs = probabilities.mean(axis=0)
+    expected_mean = 1.0 / len(np.unique(y))
+    assert np.allclose(
+        mean_probs,
+        expected_mean,
+        rtol=0.1,
+    ), "Class probabilities are not properly balanced in pipeline"
 
 
 def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     """Test that dict configs behave identically to PreprocessorConfig objects."""
     X, y = X_y
 
-    # Define same config as both dict and object
     dict_config = {
         "name": "quantile_uni_coarse",
         "append_original": False,  # changed from default
         "categorical_name": "ordinal_very_common_categories_shuffled",
         "global_transformer_name": "svd",
-        "max_features_per_estimator": 500,
+        "subsample_features": -1,
     }
 
     object_config = PreprocessorConfig(
@@ -600,10 +454,9 @@ def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray])
         append_original=False,  # changed from default
         categorical_name="ordinal_very_common_categories_shuffled",
         global_transformer_name="svd",
-        max_features_per_estimator=500,
+        subsample_features=-1,
     )
 
-    # Create two models with same random state
     model_dict = TabPFNClassifier(
         inference_config={"PREPROCESS_TRANSFORMS": [dict_config]},
         n_estimators=2,
@@ -616,16 +469,13 @@ def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray])
         random_state=42,
     )
 
-    # Fit both models
     model_dict.fit(X, y)
     model_obj.fit(X, y)
 
-    # Compare predictions
     pred_dict = model_dict.predict(X)
     pred_obj = model_obj.predict(X)
     np.testing.assert_array_equal(pred_dict, pred_obj)
 
-    # Compare probabilities
     prob_dict = model_dict.predict_proba(X)
     prob_obj = model_obj.predict_proba(X)
     np.testing.assert_array_almost_equal(prob_dict, prob_obj)
@@ -691,10 +541,12 @@ def _patch_layernorm_no_affine(model: nn.Module) -> None:
 def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     if os.name == "nt":
         pytest.skip("onnx export is not tested on windows")
+    if sys.version_info >= (3, 13):
+        pytest.xfail("onnx is not yet supported on Python 3.13")
     X, y = X_y
     with torch.no_grad():
         classifier = TabPFNClassifier(n_estimators=1, device="cpu", random_state=42)
-        # load the model so we can access it via classifier.models_
+        # load the model so we can access it via classifier.model_
         classifier.fit(X, y)
         # this is necessary if cuda is available
         classifier.predict(X)
@@ -712,14 +564,9 @@ def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
             "X": {0: "num_datapoints", 1: "batch_size", 2: "num_features"},
             "y": {0: "num_labels"},
         }
-        _patch_layernorm_no_affine(classifier.models_[0])
-
-        # From 2.9 PyTorch changed the default export mode from TorchScript to
-        # Dynamo. We don't support Dynamo, so disable it. The `dynamo` flag is only
-        # available in newer PyTorch versions, hence we don't always include it.
-        export_kwargs = {"dynamo": False} if torch.__version__ >= "2.9" else {}
+        _patch_layernorm_no_affine(classifier.model_)
         torch.onnx.export(
-            ModelWrapper(classifier.models_[0]).eval(),
+            ModelWrapper(classifier.model_).eval(),
             (X_tensor, y_tensor, True, [[]]),
             io.BytesIO(),
             input_names=[
@@ -731,7 +578,6 @@ def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
             output_names=["output"],
             opset_version=17,  # using 17 since we use torch>=2.1
             dynamic_axes=dynamic_axes,
-            **export_kwargs,
         )
 
 
@@ -749,10 +595,10 @@ def test_get_embeddings(X_y: tuple[np.ndarray, np.ndarray], data_source: str) ->
     embeddings = model.get_embeddings(X, valid_data_source)
 
     # Need to access the model through the executor
-    model_instances = typing.cast(typing.Any, model.executor_).models
+    model_instance = typing.cast(typing.Any, model.executor_).model
     encoder_shape = next(
         m.out_features
-        for m in model_instances[0].encoder.modules()
+        for m in model_instance.encoder.modules()
         if isinstance(m, nn.Linear)
     )
 
@@ -762,9 +608,14 @@ def test_get_embeddings(X_y: tuple[np.ndarray, np.ndarray], data_source: str) ->
     assert embeddings.shape[2] == encoder_shape
 
 
-def test_pandas_output_config(X_y: tuple[np.ndarray, np.ndarray]):
+def test_pandas_output_config():
     """Test compatibility with sklearn's output configuration settings."""
-    X, y = X_y
+    # Generate synthetic classification data
+    X, y = sklearn.datasets.make_classification(
+        n_samples=50,
+        n_features=10,
+        random_state=19,
+    )
 
     # Initialize TabPFN
     model = TabPFNClassifier(n_estimators=1, random_state=42)
@@ -872,7 +723,7 @@ def test_classifier_with_text_and_na() -> None:
     y = df["target"]
 
     # Initialize and fit TabPFN on data with text+NA and a column with all NAs
-    classifier = TabPFNClassifier(device="auto", n_estimators=2)
+    classifier = TabPFNClassifier(device="cpu", n_estimators=2)
 
     # This should now work without raising errors
     classifier.fit(X, y)
@@ -888,311 +739,49 @@ def test_classifier_with_text_and_na() -> None:
 
 def test_initialize_model_variables_classifier_sets_required_attributes() -> None:
     # 1) Standalone initializer
-    models, architecture_configs, norm_criterion, inference_config = (
-        initialize_tabpfn_model(
-            model_path="auto",
-            which="classifier",
-            fit_mode="low_memory",
-        )
-    )
-    assert models is not None, "model should be initialized for classifier"
-    assert architecture_configs is not None, (
-        "config should be initialized for classifier"
+    model, config, norm_criterion = initialize_tabpfn_model(
+        model_path="auto",
+        which="classifier",
+        fit_mode="low_memory",
     )
+    assert model is not None, "model should be initialized for classifier"
+    assert config is not None, "config should be initialized for classifier"
     assert norm_criterion is None, "norm_criterion should be None for classifier"
-    assert inference_config is not None
 
     # 2) Test the sklearn-style wrapper on TabPFNClassifier
-    classifier = TabPFNClassifier(device="cpu", random_state=42)
+    classifier = TabPFNClassifier(model_path="auto", device="cpu", random_state=42)
     classifier._initialize_model_variables()
 
-    assert hasattr(classifier, "models_")
-    assert classifier.models_ is not None
+    assert hasattr(classifier, "model_"), "classifier should have model_ attribute"
+    assert classifier.model_ is not None, "model_ should be initialized for classifier"
 
-    assert hasattr(classifier, "configs_")
-    assert classifier.configs_ is not None
+    assert hasattr(classifier, "config_"), "classifier should have config_ attribute"
+    assert (
+        classifier.config_ is not None
+    ), "config_ should be initialized for classifier"
 
-    assert not hasattr(classifier, "znorm_space_bardist_")
+    assert not hasattr(
+        classifier, "bardist_"
+    ), "classifier should not have bardist_ attribute"
 
     # 3) Reuse via ClassifierModelSpecs
-    spec = ClassifierModelSpecs(
-        model=classifier.models_[0],
-        architecture_config=classifier.configs_[0],
-        inference_config=classifier.inference_config_,
-    )
+    new_model_state = classifier.model_
+    new_config = classifier.config_
+    spec = ClassifierModelSpecs(model=new_model_state, config=new_config)
 
     classifier2 = TabPFNClassifier(model_path=spec)
     classifier2._initialize_model_variables()
 
-    assert hasattr(classifier2, "models_")
-    assert classifier2.models_ is not None
-
-    assert hasattr(classifier2, "configs_")
-    assert classifier2.configs_ is not None
-
-    assert not hasattr(classifier2, "znorm_space_bardist_")
-
-
-@pytest.mark.parametrize("n_features", [1, 2])
-def test__TabPFNClassifier__few_features__works(n_features: int) -> None:
-    """Test that TabPFNClassifier works correctly with 1 or 2 features."""
-    n_classes = 2
-    n_samples = 20 * n_classes
+    assert hasattr(classifier2, "model_"), "classifier2 should have model_ attribute"
+    assert (
+        classifier2.model_ is not None
+    ), "model_ should be initialized for classifier2"
 
-    X, y = sklearn.datasets.make_classification(
-        n_samples=n_samples,
-        n_classes=n_classes,
-        n_features=n_features,
-        n_informative=n_features,
-        n_redundant=0,
-        n_clusters_per_class=1,
-        random_state=42,
-    )
-
-    model = TabPFNClassifier(
-        n_estimators=2,
-        random_state=42,
-    )
-
-    returned_model = model.fit(X, y)
-    assert returned_model is model, "Returned model is not the same as the model"
-    check_is_fitted(returned_model)
-
-    probabilities = model.predict_proba(X)
-    assert probabilities.shape == (
-        X.shape[0],
-        n_classes,
-    ), f"Probabilities shape is incorrect for {n_features} features"
-    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities do not sum to 1"
-
-    predictions = model.predict(X)
-    assert predictions.shape == (X.shape[0],), (
-        f"Predictions shape is incorrect for {n_features} features"
-    )
-    accuracy = accuracy_score(y, predictions)
-    assert accuracy > 0.3, f"Accuracy too low with {n_features} features: {accuracy}"
-
-
-@pytest.mark.parametrize(
-    (
-        "eval_metric",
-        "tuning_holdout_pct",
-        "tuning_holdout_n_splits",
-        "tune_decision_thresholds",
-        "calibrate_temperature",
-        "expected_equal",
-    ),
-    [
-        (ClassifierEvalMetrics.F1, 0.1, 1, False, True, False),
-        (ClassifierEvalMetrics.ACCURACY, 0.2, 1, False, False, True),
-        (ClassifierEvalMetrics.ACCURACY, 0.7, 1, True, False, False),
-        (ClassifierEvalMetrics.F1, 0.05, 2, True, False, False),
-        (ClassifierEvalMetrics.F1, 0.2, 1, False, True, False),
-        (ClassifierEvalMetrics.BALANCED_ACCURACY, 0.1, 1, False, False, True),
-    ],
-)
-def test__fit_with_tuning_config__works_with_different_eval_metrics(
-    eval_metric: ClassifierEvalMetrics,
-    tuning_holdout_pct: float,
-    tuning_holdout_n_splits: int,
-    tune_decision_thresholds: bool,
-    calibrate_temperature: bool,
-    expected_equal: bool,
-) -> None:
-    X, y = sklearn.datasets.make_classification(
-        n_samples=MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING + 1,
-        n_classes=2,
-        n_features=2,
-        n_informative=2,
-        n_redundant=0,
-        random_state=0,
-    )
-    max_num_classes = len(np.unique(y))
-
-    if eval_metric is ClassifierEvalMetrics.ACCURACY:
-        tuning_config = ClassifierTuningConfig(
-            calibrate_temperature=calibrate_temperature,
-            tune_decision_thresholds=tune_decision_thresholds,
-            tuning_holdout_frac=tuning_holdout_pct,
-            tuning_n_folds=tuning_holdout_n_splits,
-        )
-    else:
-        # Also check parsing tuning config as dict.
-        tuning_config = {
-            "calibrate_temperature": calibrate_temperature,
-            "tune_decision_thresholds": tune_decision_thresholds,
-            "tuning_holdout_frac": tuning_holdout_pct,
-            "tuning_n_folds": tuning_holdout_n_splits,
-        }
-
-    kwargs = {
-        "fit_mode": "fit_preprocessors",
-        "eval_metric": eval_metric,
-        "n_estimators": 1,
-        "device": "cpu",
-        "inference_precision": torch.float32,
-        "random_state": 0,
-        "model_path": _create_dummy_classifier_model_specs(
-            max_num_classes=max_num_classes
-        ),
-    }
-
-    torch.random.manual_seed(0)
-    tabpfn_with_tuning = TabPFNClassifier(
-        tuning_config=tuning_config,
-        **kwargs,
-    )
-    tabpfn_with_tuning.fit(X, y)
-    preds_with_tuning = tabpfn_with_tuning.predict_proba(X[0 : X.shape[0] // 4])
-
-    assert len(preds_with_tuning) == X.shape[0] // 4
-
-    torch.random.manual_seed(0)
-    tabpfn_no_tuning = TabPFNClassifier(**kwargs)
-    tabpfn_no_tuning.fit(X, y)
-    preds_no_tuning = tabpfn_no_tuning.predict_proba(X[0 : X.shape[0] // 4])
-
-    assert np.allclose(preds_with_tuning, preds_no_tuning, atol=1e-5) == expected_equal
-
-    if calibrate_temperature:
-        assert (
-            tabpfn_with_tuning.softmax_temperature_
-            != tabpfn_no_tuning.softmax_temperature_
-        )
-    else:
-        assert (
-            tabpfn_with_tuning.softmax_temperature_
-            == tabpfn_no_tuning.softmax_temperature_
-            == tabpfn_with_tuning.softmax_temperature
-        )
-
-
-def test__logits_to_probabilities__same_as_predict_proba(
-    X_y: tuple[np.ndarray, np.ndarray],
-) -> None:
-    X, y = X_y
-    max_num_classes = len(np.unique(y))
-
-    model = TabPFNClassifier(
-        n_estimators=1,
-        random_state=42,
-        model_path=_create_dummy_classifier_model_specs(
-            max_num_classes=max_num_classes
-        ),
-    )
-    model.fit(X, y)
-
-    raw_logits = model.predict_raw_logits(X)
-    probas = model.logits_to_probabilities(raw_logits)
-
-    expected_probas = model.predict_proba(X)
-    assert np.allclose(probas, expected_probas, atol=1e-4, rtol=1e-3)
-
-
-def test__fit_with_f1_metric_without_tuning_config__warns(
-    X_y: tuple[np.ndarray, np.ndarray],
-) -> None:
-    """Test that warning is issued when F1 metric used without tuning config."""
-    X, y = X_y
-
-    clf = TabPFNClassifier(
-        eval_metric="f1",
-        tuning_config=None,
-        n_estimators=1,
-        model_path=_create_dummy_classifier_model_specs(
-            max_num_classes=len(np.unique(y))
-        ),
-    )
-
-    with pytest.warns(
-        UserWarning,
-        match=r".*haven't specified any tuning configuration.*",
-    ):
-        clf.fit(X, y)
-
-
-def test__fit_with_small_dataset_and_tuning__warns() -> None:
-    """Test that warning is issued when F1 metric used without tuning config."""
-    default_rng = np.random.default_rng(seed=42)
-    X = default_rng.random((MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING - 1, 10))
-    y = default_rng.integers(0, 2, MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING - 1)
-
-    clf = TabPFNClassifier(
-        eval_metric="f1",
-        tuning_config={
-            "tune_decision_thresholds": True,
-        },
-        n_estimators=1,
-        model_path=_create_dummy_classifier_model_specs(
-            max_num_classes=len(np.unique(y))
-        ),
-    )
-
-    with pytest.warns(
-        UserWarning,
-        match=r".*We recommend tuning only for datasets with more than.*",
-    ):
-        clf.fit(X, y)
-
-
-def test__fit_with_roc_auc_metric_with_threshold_tuning__warns(
-    X_y: tuple[np.ndarray, np.ndarray],
-) -> None:
-    """Test that warning is issued when ROC AUC metric used with threshold tuning."""
-    X, y = X_y
-
-    clf = TabPFNClassifier(
-        eval_metric="roc_auc",
-        tuning_config={
-            "tune_decision_thresholds": True,
-            "calibrate_temperature": False,
-            "tuning_holdout_frac": 0.1,
-            "tuning_n_folds": 1,
-        },
-        n_estimators=1,
-        device="cpu",
-        model_path=_create_dummy_classifier_model_specs(
-            max_num_classes=len(np.unique(y))
-        ),
-        random_state=0,
-    )
-
-    with pytest.warns(
-        UserWarning,
-        match=(
-            r".*with threshold tuning or temperature calibration "
-            r"enabled.*is independent of these tunings.*"
-        ),
-    ):
-        clf.fit(X, y)
-
-
-def test__create_default_for_version__v2__uses_correct_defaults() -> None:
-    estimator = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
-
-    assert isinstance(estimator, TabPFNClassifier)
-    assert estimator.n_estimators == 8
-    assert estimator.softmax_temperature == 0.9
-    assert isinstance(estimator.model_path, str)
-    assert "classifier" in estimator.model_path
-    assert "-v2-" in estimator.model_path
-
-
-def test__create_default_for_version__v2_5__uses_correct_defaults() -> None:
-    estimator = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
-
-    assert isinstance(estimator, TabPFNClassifier)
-    assert estimator.n_estimators == 8
-    assert estimator.softmax_temperature == 0.9
-    assert isinstance(estimator.model_path, str)
-    assert "classifier" in estimator.model_path
-    assert "-v2.5-" in estimator.model_path
-
-
-def test__create_default_for_version__passes_through_overrides() -> None:
-    estimator = TabPFNClassifier.create_default_for_version(
-        ModelVersion.V2_5, n_estimators=16
-    )
+    assert hasattr(classifier2, "config_"), "classifier2 should have config_ attribute"
+    assert (
+        classifier2.config_ is not None
+    ), "config_ should be initialized for classifier2"
 
-    assert estimator.n_estimators == 16
-    assert estimator.softmax_temperature == 0.9
+    assert not hasattr(
+        classifier2, "bardist_"
+    ), "classifier2 should not have bardist_ attribute"
diff --git a/tests/test_consistency.py b/tests/test_consistency.py
index e1dc04b..0cb9c82 100644
--- a/tests/test_consistency.py
+++ b/tests/test_consistency.py
@@ -29,10 +29,9 @@ CI Configuration:
 - Test runs on different platforms should set FORCE_CONSISTENCY_TESTS=1
 
 If you need to update reference values:
-2. Delete the reference files you want to regenerate.
-3. Run: `FORCE_CONSISTENCY_TESTS=1 pytest tests/test_consistency.py`
-4. Include the updated reference files in your PR
-5. Document the reason for the update in your PR description
+1. Run: python tests/test_consistency.py
+2. Include the updated reference files in your PR
+3. Document the reason for the update in your PR description
 
 How It Works:
 ------------
@@ -63,11 +62,10 @@ import platform
 import numpy as np
 import pytest
 import torch
-from sklearn.datasets import load_iris
 from sklearn.utils import check_random_state
 
 # mypy: ignore-errors
-from tabpfn import TabPFNClassifier, TabPFNRegressor
+from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore
 from tabpfn.settings import settings
 
 # Test configuration parameters
@@ -97,7 +95,7 @@ _METADATA_FILE = (
 )
 
 
-def _get_platform_details() -> tuple[dict[str, str], dict[str, str]]:
+def _get_platform_details():
     """Gathers and returns details for both current and reference platforms.
 
     This function centralizes platform information retrieval and file I/O,
@@ -167,7 +165,7 @@ def is_ci_compatible_platform(os_name, python_version):
     return (os_name, python_major_minor) in CI_PLATFORMS
 
 
-def _generate_skip_logic() -> tuple[bool, str]:
+def _generate_skip_logic():
     """Determines if tests should be skipped and generates the reason string.
 
     This is the core logic that replaces should_run_consistency_tests()
@@ -264,6 +262,8 @@ def get_tiny_regression_data():
 
 def get_iris_multiclass_data():
     """Get a small subset of iris data for multiclass testing."""
+    from sklearn.datasets import load_iris
+
     # Load iris dataset with 3 well-separated classes
     X, y = load_iris(return_X_y=True)
 
@@ -306,12 +306,12 @@ class ConsistencyTest:
     PLATFORM_METADATA_FILE = REFERENCE_DIR / "platform_metadata.json"
 
     @classmethod
-    def setup_class(cls) -> None:
+    def setup_class(cls):
         """Ensure the reference predictions directory exists."""
         cls.REFERENCE_DIR.mkdir(exist_ok=True)
 
     @classmethod
-    def save_platform_metadata(cls) -> None:
+    def save_platform_metadata(cls):
         """Save current platform information to metadata file."""
         metadata = {
             "os": platform.system(),
@@ -324,6 +324,22 @@ class ConsistencyTest:
         with cls.PLATFORM_METADATA_FILE.open("w") as f:
             json.dump(metadata, f, indent=2)
 
+    @classmethod
+    def load_platform_metadata(cls):
+        """Load platform metadata from file.
+
+        Returns empty dict if file doesn't exist or can't be read.
+        """
+        if not cls.PLATFORM_METADATA_FILE.exists():
+            return {}
+
+        try:
+            with cls.PLATFORM_METADATA_FILE.open("r") as f:
+                return json.load(f)
+        except (json.JSONDecodeError, OSError):
+            # More specific exceptions for file reading issues
+            return {}
+
     def get_dataset_name(self):
         """Get the unique name for this test case."""
         raise NotImplementedError("Subclasses must implement get_dataset_name()")
@@ -409,14 +425,11 @@ class ConsistencyTest:
         return predictions
 
 
-class TestTinyClassifierFitPreprocessors(ConsistencyTest):
-    """Test prediction consistency for a tiny binary classifier.
-
-    Use `fit_mode=fit_preprocessors`.
-    """
+class TestTinyClassifier(ConsistencyTest):
+    """Test prediction consistency for a tiny binary classifier."""
 
     def get_dataset_name(self):
-        return "tiny_classifier_fit_preprocessors"
+        return "tiny_classifier"
 
     def get_test_data(self):
         return get_tiny_classification_data()
@@ -425,8 +438,7 @@ class TestTinyClassifierFitPreprocessors(ConsistencyTest):
         return TabPFNClassifier(
             n_estimators=DEFAULT_N_ESTIMATORS,
             random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            fit_mode="fit_preprocessors",
+            device="cpu",
         )
 
     def get_prediction_func(self):
@@ -438,62 +450,6 @@ class TestTinyClassifierFitPreprocessors(ConsistencyTest):
         self.run_test()
 
 
-class TestTinyClassifierLowMemory(ConsistencyTest):
-    """Test prediction consistency for a tiny binary classifier.
-
-    Use `fit_mode=low_memory`.
-    """
-
-    def get_dataset_name(self):
-        return "tiny_classifier_low_memory"
-
-    def get_test_data(self):
-        return get_tiny_classification_data()
-
-    def get_model(self):
-        return TabPFNClassifier(
-            n_estimators=DEFAULT_N_ESTIMATORS,
-            random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            fit_mode="low_memory",
-        )
-
-    def get_prediction_func(self):
-        return lambda model, X: model.predict_proba(X)
-
-    @platform_specific
-    def test_consistency(self):
-        self.run_test()
-
-
-class TestTinyClassifierFitWithCache(ConsistencyTest):
-    """Test prediction consistency for a tiny binary classifier.
-
-    Use `fit_mode=fit_with_cache`.
-    """
-
-    def get_dataset_name(self):
-        return "tiny_classifier_fit_with_cache"
-
-    def get_test_data(self):
-        return get_tiny_classification_data()
-
-    def get_model(self):
-        return TabPFNClassifier(
-            n_estimators=DEFAULT_N_ESTIMATORS,
-            random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            fit_mode="fit_with_cache",
-        )
-
-    def get_prediction_func(self):
-        return lambda model, X: model.predict_proba(X)
-
-    @platform_specific
-    def test_consistency(self):
-        self.run_test()
-
-
 class TestTinyClassifierDifferentiableInput(ConsistencyTest):
     """Test prediction consistency for a tiny binary classifier."""
 
@@ -507,7 +463,7 @@ class TestTinyClassifierDifferentiableInput(ConsistencyTest):
         return TabPFNClassifier(
             n_estimators=DEFAULT_N_ESTIMATORS,
             random_state=FIXED_RANDOM_SEED,
-            device="auto",
+            device="cpu",
             differentiable_input=True,
         )
 
@@ -522,16 +478,11 @@ class TestTinyClassifierDifferentiableInput(ConsistencyTest):
         self.run_test()
 
 
-class TestTinyRegressorFitPreprocessors(ConsistencyTest):
-    """Test prediction consistency for a tiny regressor.
-
-    Use `fit_mode=fit_preprocessors`.
-    """
-
-    DATASET_NAME = "tiny_regressor_fit_preprocessors"
+class TestTinyRegressor(ConsistencyTest):
+    """Test prediction consistency for a tiny regressor."""
 
     def get_dataset_name(self):
-        return self.DATASET_NAME
+        return "tiny_regressor"
 
     def get_test_data(self):
         return get_tiny_regression_data()
@@ -540,102 +491,8 @@ class TestTinyRegressorFitPreprocessors(ConsistencyTest):
         return TabPFNRegressor(
             n_estimators=DEFAULT_N_ESTIMATORS,
             random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            fit_mode="fit_preprocessors",
-        )
-
-    def get_prediction_func(self):
-        return lambda model, X: model.predict(X)
-
-    @platform_specific
-    def test_consistency(self):
-        """Test prediction consistency on a very small regression dataset."""
-        self.run_test()
-
-
-class TestTinyRegressorLowMemory(ConsistencyTest):
-    """Test prediction consistency for a tiny regressor.
-
-    Use `fit_mode=low_memory`.
-    """
-
-    def get_dataset_name(self):
-        return "tiny_regressor_low_memory"
-
-    def get_test_data(self):
-        return get_tiny_regression_data()
-
-    def get_model(self):
-        return TabPFNRegressor(
-            n_estimators=DEFAULT_N_ESTIMATORS,
-            random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            fit_mode="low_memory",
-        )
-
-    def get_prediction_func(self):
-        return lambda model, X: model.predict(X)
-
-    @platform_specific
-    def test_consistency(self):
-        """Test prediction consistency on a very small regression dataset."""
-        self.run_test()
-
-
-class TestTinyRegressorFitWithCache(ConsistencyTest):
-    """Test prediction consistency for a tiny regressor.
-
-    Use `fit_mode=fit_with_cache`.
-    """
-
-    def get_dataset_name(self):
-        return "tiny_regressor_fit_with_cache"
-
-    def get_test_data(self):
-        return get_tiny_regression_data()
-
-    def get_model(self):
-        return TabPFNRegressor(
-            n_estimators=DEFAULT_N_ESTIMATORS,
-            random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            fit_mode="fit_with_cache",
-        )
-
-    def get_prediction_func(self):
-        return lambda model, X: model.predict(X)
-
-    @platform_specific
-    def test_consistency(self):
-        """Test prediction consistency on a very small regression dataset."""
-        self.run_test()
-
-
-class TestTinyRegressorSeveralDevices(ConsistencyTest):
-    """Test a regressor with several CPU devices, to simulate multi-device inference."""
-
-    def get_dataset_name(self):
-        # Use the same name as TestTinyRegressorFitPreprocessors so that the predictions
-        # of the two configurations are compared to each other: multi-device inference
-        # should not affect the result.
-        return TestTinyRegressorFitPreprocessors.DATASET_NAME
-
-    def get_test_data(self):
-        return get_tiny_regression_data()
-
-    def get_model(self):
-        regressor = TabPFNRegressor(
-            n_estimators=DEFAULT_N_ESTIMATORS,
-            random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            # We select a fit mode that supports multi-device inference.
-            fit_mode="fit_preprocessors",
+            device="cpu",
         )
-        # The regressor does not allow specifying the same device twice, so override
-        # after the fact instead.
-        # Use lots of devices to maximise the chance of hitting a race condition.
-        regressor.devices_ = (torch.device("cpu"),) * DEFAULT_N_ESTIMATORS
-        return regressor
 
     def get_prediction_func(self):
         return lambda model, X: model.predict(X)
@@ -649,10 +506,8 @@ class TestTinyRegressorSeveralDevices(ConsistencyTest):
 class TestMulticlassClassifier(ConsistencyTest):
     """Test prediction consistency for a multiclass classifier."""
 
-    DATASET_NAME = "iris_multiclass"
-
     def get_dataset_name(self):
-        return self.DATASET_NAME
+        return "iris_multiclass"
 
     def get_test_data(self):
         return get_iris_multiclass_data()
@@ -661,43 +516,8 @@ class TestMulticlassClassifier(ConsistencyTest):
         return TabPFNClassifier(
             n_estimators=DEFAULT_N_ESTIMATORS,
             random_state=FIXED_RANDOM_SEED,
-            device="auto",
-        )
-
-    def get_prediction_func(self):
-        return lambda model, X: model.predict_proba(X)
-
-    @platform_specific
-    def test_consistency(self):
-        """Test prediction consistency on iris multiclass dataset."""
-        self.run_test()
-
-
-class TestMulticlassClassifierSeveralDevices(ConsistencyTest):
-    """Test a classifier on several CPU devices, to simulate multi-device inference."""
-
-    def get_dataset_name(self):
-        # Use the same name as TestMulticlassClassifier so that the predictions of the
-        # two configurations are compared to each other: multi-device inference should
-        # not affect the result.
-        return TestMulticlassClassifier.DATASET_NAME
-
-    def get_test_data(self):
-        return get_iris_multiclass_data()
-
-    def get_model(self):
-        classifier = TabPFNClassifier(
-            n_estimators=DEFAULT_N_ESTIMATORS,
-            random_state=FIXED_RANDOM_SEED,
-            device="auto",
-            # We select a fit mode that supports multi-device inference.
-            fit_mode="fit_preprocessors",
+            device="cpu",
         )
-        # The classifier does not allow specifying the same device twice, so override it
-        # after the fact.
-        # Use lots of devices to maximise the chance of hitting a race condition.
-        classifier.devices_ = (torch.device("cpu"),) * DEFAULT_N_ESTIMATORS
-        return classifier
 
     def get_prediction_func(self):
         return lambda model, X: model.predict_proba(X)
@@ -721,7 +541,7 @@ class TestEnsembleClassifier(ConsistencyTest):
         return TabPFNClassifier(
             n_estimators=5,  # Larger ensemble for this test
             random_state=FIXED_RANDOM_SEED,
-            device="auto",
+            device="cpu",
         )
 
     def get_prediction_func(self):
diff --git a/tests/test_download_fallbacks.py b/tests/test_download_fallbacks.py
deleted file mode 100644
index cd9e02e..0000000
--- a/tests/test_download_fallbacks.py
+++ /dev/null
@@ -1,49 +0,0 @@
-from __future__ import annotations
-
-import urllib.error
-import urllib.request
-from pathlib import Path
-from unittest.mock import patch
-
-from tabpfn.model_loading import (
-    FALLBACK_S3_BASE_URL,
-    ModelSource,
-    _try_direct_downloads,
-)
-
-
-class DummyResponse:
-    """Simple context manager mimicking ``urllib`` responses."""
-
-    def __init__(self, status: int = 200, data: bytes = b"ok") -> None:
-        """Create a dummy response with a given status and payload."""
-        self.status = status
-        self._data = data
-
-    def read(self) -> bytes:
-        return self._data
-
-    def __enter__(self):
-        return self
-
-    def __exit__(self, exc_type, exc, tb):
-        pass
-
-
-def test_direct_download_fallback(tmp_path: Path):
-    src = ModelSource.get_classifier_v2()
-    base_path = tmp_path / src.default_filename
-
-    attempted_urls: list[str] = []
-
-    def fake_urlopen(url: str, *_args, **_kwargs) -> DummyResponse:
-        attempted_urls.append(url)
-        if "huggingface.co" in url:
-            raise urllib.error.URLError("HF down")
-        return DummyResponse()
-
-    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
-        _try_direct_downloads(base_path, src)
-
-    assert any(url.startswith("https://huggingface.co") for url in attempted_urls)
-    assert any(url.startswith(FALLBACK_S3_BASE_URL) for url in attempted_urls)
diff --git a/tests/test_finetuning_classifier.py b/tests/test_finetuning_classifier.py
index 0362c07..bbcd0d1 100644
--- a/tests/test_finetuning_classifier.py
+++ b/tests/test_finetuning_classifier.py
@@ -20,11 +20,11 @@ from tabpfn.preprocessing import (
 )
 from tabpfn.utils import meta_dataset_collator
 
-from .utils import get_pytest_devices
-
 rng = np.random.default_rng(42)
 
-devices = get_pytest_devices()
+devices = ["cpu"]
+if torch.cuda.is_available():
+    devices.append("cuda")
 
 fit_modes = [
     "batched",
@@ -46,7 +46,7 @@ param_order = [
 
 default_config = {
     "n_estimators": 1,
-    "device": "auto",
+    "device": "cpu",
     "fit_mode": "batched",
     "inference_precision": "auto",
 }
@@ -174,7 +174,7 @@ def test_tabpfn_classifier_finetuning_loop(
     synthetic_data,
 ) -> None:
     X, y = synthetic_data
-    X_train, _X_test, y_train, _y_test = train_test_split(
+    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=42
     )
 
@@ -244,13 +244,13 @@ def test_tabpfn_classifier_finetuning_loop(
             assert torch.isfinite(loss).all(), f"Loss is not finite: {loss.item()}"
 
             # --- Gradient Check ---
-            assert hasattr(clf, "models_")
-            assert clf.models_ is not None
-            clf.models_[0].zero_grad()
+            assert hasattr(clf, "model_"), "Classifier missing 'model_'"
+            assert clf.model_ is not None, "Classifier model is None"
+            clf.model_.zero_grad()
             loss.backward()
 
             gradients_found = False
-            for param in clf.models_[0].parameters():
+            for param in clf.model_.parameters():
                 if (
                     param.requires_grad
                     and param.grad is not None
@@ -297,17 +297,17 @@ def test_datasetcollectionwithpreprocessing_classification_single_dataset(
     processed_dataset_item = dataset_collection[item_index]
 
     assert isinstance(processed_dataset_item, tuple)
-    assert len(processed_dataset_item) == 6, (
-        "Item tuple should have 4 elements for classification"
-    )
+    assert (
+        len(processed_dataset_item) == 6
+    ), "Item tuple should have 4 elements for classification"
 
     (
         X_trains_preprocessed,
-        _X_tests_preprocessed,
-        _y_trains_preprocessed,
+        X_tests_preprocessed,
+        y_trains_preprocessed,
         y_test_raw_tensor,
-        _cat_ixs,
-        _returned_ensemble_configs,
+        cat_ixs,
+        returned_ensemble_configs,
     ) = processed_dataset_item
 
     assert isinstance(X_trains_preprocessed, list)
@@ -339,23 +339,23 @@ def test_datasetcollectionwithpreprocessing_classification_multiple_datasets(
     )
 
     assert isinstance(dataset_collection, DatasetCollectionWithPreprocessing)
-    assert len(dataset_collection) == len(datasets), (
-        "Collection should contain one item per dataset"
-    )
+    assert len(dataset_collection) == len(
+        datasets
+    ), "Collection should contain one item per dataset"
 
     for item_index in range(len(datasets)):
         processed_dataset_item = dataset_collection[item_index]
         assert isinstance(processed_dataset_item, tuple)
-        assert len(processed_dataset_item) == 6, (
-            "Item tuple should have 6 elements for classification"
-        )
+        assert (
+            len(processed_dataset_item) == 6
+        ), "Item tuple should have 6 elements for classification"
         (
             X_trains_preprocessed,
-            _X_tests_preprocessed,
-            _y_trains_preprocessed,
+            X_tests_preprocessed,
+            y_trains_preprocessed,
             y_test_raw_tensor,
-            _cat_ixs,
-            _returned_ensemble_configs,
+            cat_ixs,
+            returned_ensemble_configs,
         ) = processed_dataset_item
         assert isinstance(X_trains_preprocessed, list)
         assert len(X_trains_preprocessed) == n_estimators
@@ -383,16 +383,16 @@ def test_dataset_and_collator_with_dataloader_uniform(
     for batch in dl:
         # Should be tuple with X_trains, X_tests, y_trains, y_tests, cat_ixs, confs
         assert isinstance(batch, tuple)
-        X_trains, _X_tests, y_trains, _y_tests, _cat_ixs, _confs = batch
+        X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = batch
         for est_tensor in X_trains:
-            assert isinstance(est_tensor, torch.Tensor), (
-                "Each estimator's batch should be a tensor."
-            )
+            assert isinstance(
+                est_tensor, torch.Tensor
+            ), "Each estimator's batch should be a tensor."
             assert est_tensor.shape[0] == batch_size
         for est_tensor in y_trains:
-            assert isinstance(est_tensor, torch.Tensor), (
-                "Each estimator's batch should be a tensor for labels."
-            )
+            assert isinstance(
+                est_tensor, torch.Tensor
+            ), "Each estimator's batch should be a tensor for labels."
             assert est_tensor.shape[0] == batch_size
         break  # Only check one batch
 
@@ -416,7 +416,7 @@ def test_classifier_dataset_and_collator_batches_type(
     )
     for batch in dl:
         assert isinstance(batch, tuple)
-        X_trains, _X_tests, y_trains, _y_tests, cat_ixs, confs = batch
+        X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = batch
         for est_tensor in X_trains:
             assert isinstance(est_tensor, torch.Tensor)
             assert est_tensor.shape[0] == batch_size
@@ -457,7 +457,7 @@ def test_get_preprocessed_datasets_categorical_features(classifier_instance):
 
 def test_forward_runs(classifier_instance, classification_data):
     """Ensure predict_proba_tensor runs OK after standard fit."""
-    X_train, X_test, y_train, _y_test = classification_data
+    X_train, X_test, y_train, y_test = classification_data
     clf = classifier_instance
     clf.fit_mode = "low_memory"
     clf.fit(X_train, y_train)
@@ -469,9 +469,9 @@ def test_forward_runs(classifier_instance, classification_data):
     assert preds.shape[0] == X_test.shape[0], "Mismatch in test sample count"
     assert preds.shape[1] == clf.n_classes_, "Mismatch in class count"
     probs_sum = preds.sum(dim=1)
-    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), (
-        "Probabilities do not sum to 1"
-    )
+    assert torch.allclose(
+        probs_sum, torch.ones_like(probs_sum), atol=1e-5
+    ), "Probabilities do not sum to 1"
 
 
 def test_fit_from_preprocessed_runs(classifier_instance, classification_data) -> None:
@@ -479,7 +479,7 @@ def test_fit_from_preprocessed_runs(classifier_instance, classification_data) ->
     using prepared data and produces
     valid predictions.
     """
-    X_train, _X_test, y_train, _y_test = classification_data
+    X_train, X_test, y_train, y_test = classification_data
     clf = classifier_instance
 
     split_fn = partial(train_test_split, test_size=0.3, random_state=42)
@@ -501,9 +501,9 @@ def test_fit_from_preprocessed_runs(classifier_instance, classification_data) ->
 
         # TODO: verify number of classes, "Mismatch in class count"
         probs_sum = preds.sum(dim=1)
-        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), (
-            "Probabilities do not sum to 1"
-        )
+        assert torch.allclose(
+            probs_sum, torch.ones_like(probs_sum), atol=1e-5
+        ), "Probabilities do not sum to 1"
         break  # Only need to check one batch for this test
 
 
@@ -540,12 +540,12 @@ class TestTabPFNClassifierPreprocessingInspection(unittest.TestCase):
             random_state=common_seed,
             shuffle=False,  # Keep False for consistent splitting
         )
-        X_train_raw, X_test_raw, y_train_raw, _y_test_raw = splitfn(X, y)
+        X_train_raw, X_test_raw, y_train_raw, y_test_raw = splitfn(X, y)
 
         # Initialize two classifiers with the necessary modes
         clf_standard = TabPFNClassifier(
             n_estimators=n_estimators,
-            device="auto",
+            device="cpu",
             random_state=common_seed,
             fit_mode="fit_preprocessors",  # A standard mode that preprocesses on fit
         )
@@ -553,7 +553,7 @@ class TestTabPFNClassifierPreprocessingInspection(unittest.TestCase):
         #  and fit_from_preprocessed
         clf_batched = TabPFNClassifier(
             n_estimators=n_estimators,
-            device="auto",
+            device="cpu",
             random_state=common_seed,
             fit_mode="batched",
         )
@@ -563,39 +563,36 @@ class TestTabPFNClassifierPreprocessingInspection(unittest.TestCase):
         clf_standard.fit(X_train_raw, y_train_raw)
         # Ensure the internal model attribute exists after fit
         assert all(
-            [
-                hasattr(clf_standard, "models_"),
-                hasattr(clf_standard.models_[0], "forward"),
-            ]
-        ), "Standard classifier models_ or models_[0].forward not found after fit."
+            [hasattr(clf_standard, "model_"), hasattr(clf_standard.model_, "forward")]
+        ), "Standard classifier model_ or model_.forward not found after fit."
 
         tensor_p1_full = None
         # Patch the standard classifier's *internal model's* forward method
         # The internal model typically receives the combined train+test sequence
         with patch.object(
-            clf_standard.models_[0], "forward", wraps=clf_standard.models_[0].forward
+            clf_standard.model_, "forward", wraps=clf_standard.model_.forward
         ) as mock_forward_p1:
             _ = clf_standard.predict_proba(X_test_raw)
-            assert mock_forward_p1.called, "Standard models_[0].forward was not called."
+            assert mock_forward_p1.called, "Standard model_.forward was not called."
 
             # Capture the tensor input 'x' (usually the second positional argument)
             call_args_list = mock_forward_p1.call_args_list
-            assert len(call_args_list) > 0, (
-                "No calls recorded for standard models_[0].forward."
-            )
+            assert (
+                len(call_args_list) > 0
+            ), "No calls recorded for standard model_.forward."
             if len(call_args_list[0].args) > 1:
                 tensor_p1_full = call_args_list[0].args[0]
                 tensor_p1_full = mock_forward_p1.call_args.args[0]
 
             else:
                 self.fail(
-                    f"Standard models_[0].forward call had "
+                    f"Standard model_.forward call had "
                     f"unexpected arguments: {call_args_list[0].args}"
                 )
 
-        assert tensor_p1_full is not None, (
-            "Failed to capture tensor from standard path."
-        )
+        assert (
+            tensor_p1_full is not None
+        ), "Failed to capture tensor from standard path."
         # Shape might be [1, N_Total, Features+1] or similar. Check the actual shape.
         # Example assertion: Check if the sequence length matches n_total
         assert tensor_p1_full.shape[0] == n_total, (
@@ -639,12 +636,9 @@ class TestTabPFNClassifierPreprocessingInspection(unittest.TestCase):
             X_trains_p2, y_trains_p2, cat_ixs_p2, confs_p2
         )
         assert all(
-            [
-                hasattr(clf_batched, "models_"),
-                hasattr(clf_batched.models_[0], "forward"),
-            ]
+            [hasattr(clf_batched, "model_"), hasattr(clf_batched.model_, "forward")]
         ), (
-            "Batched classifier models_ or models_[0].forward not"
+            "Batched classifier model_ or model_.forward not"
             "found after fit_from_preprocessed."
         )
 
@@ -653,21 +647,21 @@ class TestTabPFNClassifierPreprocessingInspection(unittest.TestCase):
         tensor_p2_full = None
         # Patch the *batched* classifier's internal model's forward method
         with patch.object(
-            clf_batched.models_[0], "forward", wraps=clf_batched.models_[0].forward
+            clf_batched.model_, "forward", wraps=clf_batched.model_.forward
         ) as mock_forward_p2:
             _ = clf_batched.forward(X_tests_p2)
-            assert mock_forward_p2.called, "Batched models_[0].forward was not called."
+            assert mock_forward_p2.called, "Batched model_.forward was not called."
 
             # Capture the tensor input 'x' (assuming same argument position as Path 1)
             call_args_list = mock_forward_p2.call_args_list
-            assert len(call_args_list) > 0, (
-                "No calls recorded for batched models_[0].forward."
-            )
+            assert (
+                len(call_args_list) > 0
+            ), "No calls recorded for batched model_.forward."
             if len(call_args_list[0].args) > 1:
                 tensor_p2_full = mock_forward_p2.call_args.args[0]
             else:
                 self.fail(
-                    f"Batched models_[0].forward call had "
+                    f"Batched model_.forward call had "
                     f"unexpected arguments: {call_args_list[0].args}"
                 )
 
@@ -699,9 +693,9 @@ class TestTabPFNClassifierPreprocessingInspection(unittest.TestCase):
             p2_squeezed = tensor_p2_full
 
         # Final check of shapes after potential squeeze
-        assert p1_squeezed.shape == p2_squeezed.shape, (
-            "Shapes of final model input tensors mismatch after squeeze. "
-        )
+        assert (
+            p1_squeezed.shape == p2_squeezed.shape
+        ), "Shapes of final model input tensors mismatch after squeeze. "
 
         # Visual inspection snippet
 
@@ -715,7 +709,7 @@ class TestTabPFNClassifierPreprocessingInspection(unittest.TestCase):
         if not tensors_match:
             diff = torch.abs(p1_squeezed - p2_squeezed)
             # Find where they differ most
-            _max_diff_val, max_diff_idx = torch.max(diff.flatten(), dim=0)
+            max_diff_val, max_diff_idx = torch.max(diff.flatten(), dim=0)
             np.unravel_index(max_diff_idx.item(), p1_squeezed.shape)
 
         # Assertion: The final tensors fed to the model sh
diff --git a/tests/test_finetuning_regressor.py b/tests/test_finetuning_regressor.py
index 780611e..7b2945d 100644
--- a/tests/test_finetuning_regressor.py
+++ b/tests/test_finetuning_regressor.py
@@ -2,7 +2,7 @@ from __future__ import annotations
 
 import unittest
 from functools import partial
-from typing import Literal
+from typing import Any, Literal
 from unittest.mock import patch
 
 import numpy as np
@@ -21,11 +21,11 @@ from tabpfn.architectures.base.bar_distribution import (
 from tabpfn.preprocessing import RegressorEnsembleConfig
 from tabpfn.utils import meta_dataset_collator
 
-from .utils import get_pytest_devices
-
 rng = np.random.default_rng(42)
 
-devices = get_pytest_devices()
+devices = ["cpu"]
+if torch.cuda.is_available():
+    devices.append("cuda")
 
 fit_modes = [
     "batched",
@@ -54,7 +54,7 @@ default_config = {
     "optimization_space": "raw_label_space",
 }
 
-param_values: dict[str, list] = {
+param_values: dict[str, list[Any]] = {
     "n_estimators": estimators,
     "device": devices,
     "fit_mode": fit_modes,
@@ -162,15 +162,15 @@ def test_regressor_dataset_and_collator_batches_type(
         assert isinstance(batch, tuple)
         (
             X_trains_preprocessed,
-            _X_tests_preprocessed,
+            X_tests_preprocessed,
             y_trains_preprocessed,
-            _y_test_standardized,
+            y_test_standardized,
             cat_ixs,
             confs,
-            raw_space_bardist_,
+            normalized_bardist_,
             bar_distribution,
-            _x_test_raw,
-            _y_test_raw,
+            x_test_raw,
+            y_test_raw,
         ) = batch
         for est_tensor in X_trains_preprocessed:
             assert isinstance(est_tensor, torch.Tensor)
@@ -182,7 +182,7 @@ def test_regressor_dataset_and_collator_batches_type(
         for conf in confs:
             for c in conf:
                 assert isinstance(c, RegressorEnsembleConfig)
-        for ren_crit in raw_space_bardist_:
+        for ren_crit in normalized_bardist_:
             assert isinstance(ren_crit, FullSupportBarDistribution)
         for bar_dist in bar_distribution:
             assert isinstance(bar_dist, BarDistribution)
@@ -199,7 +199,7 @@ def test_tabpfn_regressor_finetuning_loop(
     synthetic_regression_data,
 ) -> None:
     X, y = synthetic_regression_data
-    X_train, _X_test, y_train, _y_test = train_test_split(
+    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=42
     )
 
@@ -221,7 +221,7 @@ def test_tabpfn_regressor_finetuning_loop(
         datasets_list, batch_size=batch_size, collate_fn=meta_dataset_collator
     )
 
-    optim_impl = Adam(reg.models_[0].parameters(), lr=1e-5)
+    optim_impl = Adam(reg.model_.parameters(), lr=1e-5)
 
     if inference_precision == torch.float64:
         pass
@@ -245,9 +245,9 @@ def test_tabpfn_regressor_finetuning_loop(
                 y_test_standardized,
                 cat_ixs,
                 confs,
-                raw_space_bardist_,
-                _bar_distribution,
-                _batch_x_test_raw,
+                normalized_bardist_,
+                bar_distribution,
+                batch_x_test_raw,
                 batch_y_test_raw,
             ) = data_batch
 
@@ -255,14 +255,14 @@ def test_tabpfn_regressor_finetuning_loop(
                 X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
             )
 
-            reg.raw_space_bardist_ = raw_space_bardist_[0]
+            reg.normalized_bardist_ = normalized_bardist_[0]
 
             averaged_pred_logits, _, _ = reg.forward(X_tests_preprocessed)
 
             # --- Basic Shape Checks ---
-            assert averaged_pred_logits.ndim == 3, (
-                f"Expected 3D output, got {averaged_pred_logits.shape}"
-            )
+            assert (
+                averaged_pred_logits.ndim == 3
+            ), f"Expected 3D output, got {averaged_pred_logits.shape}"
 
             # Batch Size
             assert averaged_pred_logits.shape[0] == batch_y_test_raw.shape[0]
@@ -275,30 +275,28 @@ def test_tabpfn_regressor_finetuning_loop(
             assert averaged_pred_logits.shape[1] == y_test_standardized.shape[1]
 
             # N_bins
-            n_borders_bardist = reg.znorm_space_bardist_.borders.shape[0]
+            n_borders_bardist = reg.bardist_.borders.shape[0]
             assert averaged_pred_logits.shape[2] == n_borders_bardist - 1
-            n_borders_norm_crit = reg.raw_space_bardist_.borders.shape[0]
+            n_borders_norm_crit = reg.normalized_bardist_.borders.shape[0]
             assert averaged_pred_logits.shape[2] == n_borders_norm_crit - 1
 
             assert len(X_tests_preprocessed) == reg.n_estimators
             assert len(X_trains_preprocessed) == reg.n_estimators
             assert len(y_trains_preprocessed) == reg.n_estimators
-            assert reg.models_ is not None, "Model not initialized after fit"
-            assert hasattr(reg, "znorm_space_bardist_"), (
-                "Regressor missing 'znorm_space_bardist_' attribute after fit"
-            )
-            assert hasattr(reg, "raw_space_bardist_"), (
-                "Regressor missing 'raw_space_bardist_' attribute after fit"
-            )
-            assert reg.znorm_space_bardist_ is not None, (
-                "reg.znorm_space_bardist_ is None"
-            )
+            assert reg.model_ is not None, "Model not initialized after fit"
+            assert hasattr(
+                reg, "bardist_"
+            ), "Regressor missing 'bardist_' attribute after fit"
+            assert hasattr(
+                reg, "normalized_bardist_"
+            ), "Regressor missing 'normalized_bardist_' attribute after fit"
+            assert reg.bardist_ is not None, "reg.bardist_ is None"
 
             lossfn = None
             if optimization_space == "raw_label_space":
-                lossfn = reg.raw_space_bardist_
+                lossfn = reg.bardist_
             elif optimization_space == "preprocessed":
-                lossfn = reg.znorm_space_bardist_
+                lossfn = reg.normalized_bardist_
             else:
                 raise ValueError("Need to define optimization space")
 
@@ -314,7 +312,7 @@ def test_tabpfn_regressor_finetuning_loop(
             assert torch.isfinite(loss).all(), f"Loss is not finite: {loss.item()}"
 
             gradients_found = False
-            for param in reg.models_[0].parameters():
+            for param in reg.model_.parameters():
                 if (
                     param.requires_grad
                     and param.grad is not None
@@ -324,7 +322,7 @@ def test_tabpfn_regressor_finetuning_loop(
                     break
             assert gradients_found, "No non-zero gradients found."
 
-            reg.models_[0].zero_grad()
+            reg.model_.zero_grad()
             break  # Only test one batch
 
 
@@ -372,13 +370,13 @@ def test_finetuning_consistency_bar_distribution(
     data_batch = next(iter(dataloader))
     (
         X_trains_preprocessed,
-        _X_tests_preprocessed,
+        X_tests_preprocessed,
         y_trains_preprocessed,
         y_test_standardized,
         cat_ixs,
         confs,
-        raw_space_bardist_,
-        _bar_distribution,
+        normalized_bardist_,
+        bar_distribution,
         batch_x_test_raw,
         batch_y_test_raw,
     ) = data_batch
@@ -414,36 +412,36 @@ def test_finetuning_consistency_bar_distribution(
         atol=1e-5,
     )
 
-    raw_space_bardist_ = raw_space_bardist_[0]
-    reg_batched.raw_space_bardist_ = raw_space_bardist_
+    normalized_bardist_ = normalized_bardist_[0]
+    reg_batched.normalized_bardist_ = normalized_bardist_
 
     torch.testing.assert_close(
-        raw_space_bardist_.borders,
-        reg_batched.raw_space_bardist_.borders,
+        normalized_bardist_.borders,
+        reg_batched.normalized_bardist_.borders,
         rtol=1e-5,
         atol=1e-5,
         msg="Renormalized criterion borders do not match.",
     )
 
     torch.testing.assert_close(
-        raw_space_bardist_.borders,
-        reg_standard.raw_space_bardist_.borders,
+        normalized_bardist_.borders,
+        reg_standard.normalized_bardist_.borders,
         rtol=1e-5,  # Standard float tolerance
         atol=1e-5,
         msg="Renormalized criterion borders do not match.",
     )
 
     torch.testing.assert_close(
-        reg_standard.raw_space_bardist_.borders,
-        reg_batched.raw_space_bardist_.borders,
+        reg_standard.normalized_bardist_.borders,
+        reg_batched.normalized_bardist_.borders,
         rtol=1e-5,  # Standard float tolerance
         atol=1e-5,
         msg="Renormalized criterion borders do not match.",
     )
 
     torch.testing.assert_close(
-        reg_standard.znorm_space_bardist_.borders,
-        reg_batched.znorm_space_bardist_.borders,
+        reg_standard.bardist_.borders,
+        reg_batched.bardist_.borders,
         rtol=1e-5,  # Standard float tolerance
         atol=1e-5,
         msg="Bar distribution borders do not match.",
@@ -481,26 +479,26 @@ class TestTabPFNPreprocessingInspection(unittest.TestCase):
         # Initialize two regressors with the inference and FineTuning
         reg_standard = TabPFNRegressor(
             n_estimators=n_estimators,
-            device="auto",
+            device="cpu",
             random_state=common_seed,
             fit_mode="fit_preprocessors",  # Example standard mode
         )
         reg_batched = TabPFNRegressor(
             n_estimators=n_estimators,
-            device="auto",
+            device="cpu",
             random_state=common_seed,
             fit_mode="batched",  # Mode compatible with get_preprocessed_datasets
         )
 
         # --- 2. Path 1: Standard fit -> predict -> Capture Tensor ---
         reg_standard.fit(X_train_raw, y_train_raw)
-        assert hasattr(reg_standard, "models_")
-        assert hasattr(reg_standard.models_[0], "forward")
+        assert hasattr(reg_standard, "model_")
+        assert hasattr(reg_standard.model_, "forward")
 
         tensor_p1_full = None
         # Patch the standard regressor's internal model's forward method
         with patch.object(
-            reg_standard.models_[0], "forward", wraps=reg_standard.models_[0].forward
+            reg_standard.model_, "forward", wraps=reg_standard.model_.forward
         ) as mock_forward_p1:
             _ = reg_standard.predict(X_test_raw)  # Trigger the patched method
             assert mock_forward_p1.called
@@ -541,14 +539,14 @@ class TestTabPFNPreprocessingInspection(unittest.TestCase):
         reg_batched.fit_from_preprocessed(
             X_trains_p2, y_trains_p2, cat_ixs_p2, confs_p2
         )
-        assert hasattr(reg_batched, "models_")
-        assert hasattr(reg_batched.models_[0], "forward")
+        assert hasattr(reg_batched, "model_")
+        assert hasattr(reg_batched.model_, "forward")
 
         # Step 3c: Call forward and capture the input tensor to the *internal model*
         tensor_p3_full = None
         # Patch the *batched* regressor's internal model's forward method
         with patch.object(
-            reg_batched.models_[0], "forward", wraps=reg_batched.models_[0].forward
+            reg_batched.model_, "forward", wraps=reg_batched.model_.forward
         ) as mock_forward_p3:
             # Pass the list of preprocessed test tensors obtained earlier
             _ = reg_batched.forward(X_tests_p2)
@@ -563,15 +561,15 @@ class TestTabPFNPreprocessingInspection(unittest.TestCase):
 
         # --- 4. Comparison (Path 1 vs Path 3) ---
 
-        # Compare the two full tensors captured from the input to models_[0].forward
+        # Compare the two full tensors captured from the input to model_.forward
         # Squeeze dimensions of size 1 for direct comparison
         # shapes should be [N_Total, Features+1]
         p1_squeezed = tensor_p1_full.squeeze()
         p3_squeezed = tensor_p3_full.squeeze()
 
-        assert p1_squeezed.shape == p3_squeezed.shape, (
-            "Shapes of final model input tensors mismatch."
-        )
+        assert (
+            p1_squeezed.shape == p3_squeezed.shape
+        ), "Shapes of final model input tensors mismatch."
 
         atol = 1e-6
         tensors_match = torch.allclose(p1_squeezed, p3_squeezed, atol=atol)
diff --git a/tests/test_ft_utils.py b/tests/test_ft_utils.py
index 825d3f0..532b6af 100644
--- a/tests/test_ft_utils.py
+++ b/tests/test_ft_utils.py
@@ -14,17 +14,17 @@ def test_pad_tensors_2d_and_1d():
     # 2D tensors (features)
     tensors_2d = [torch.ones((2, 3)), torch.ones((3, 2)), torch.ones((1, 4))]
     padded = pad_tensors(tensors_2d, padding_val=-1, labels=False)
-    assert all(t.shape == (3, 4) for t in padded), (
-        f"Expected shape (3, 4), got {[t.shape for t in padded]}"
-    )
+    assert all(
+        t.shape == (3, 4) for t in padded
+    ), f"Expected shape (3, 4), got {[t.shape for t in padded]}"
     assert padded[0][2, 3] == -1, "Padding value not set correctly for 2D case."
 
     # 1D tensors (labels)
     tensors_1d = [torch.arange(3), torch.arange(5), torch.arange(2)]
     padded_1d = pad_tensors(tensors_1d, padding_val=99, labels=True)
-    assert all(t.shape == (5,) for t in padded_1d), (
-        f"Expected shape (5,), got {[t.shape for t in padded_1d]}"
-    )
+    assert all(
+        t.shape == (5,) for t in padded_1d
+    ), f"Expected shape (5,), got {[t.shape for t in padded_1d]}"
     assert padded_1d[0][3] == 99, "Padding value not set correctly for 1D case."
 
 
@@ -34,13 +34,10 @@ def test_split_large_data():
     max_chunk = 100
     large_x = np.arange(total_size * 2).reshape((total_size, 2))
     large_y = np.arange(total_size)
-    equal_split_size = True
 
     expected_num_chunks = ((total_size - 1) // max_chunk) + 1
 
-    x_chunks, y_chunks = split_large_data(
-        large_x, large_y, max_chunk, equal_split_size=equal_split_size
-    )
+    x_chunks, y_chunks = split_large_data(large_x, large_y, max_chunk)
 
     assert len(x_chunks) == expected_num_chunks, "Incorrect X chunk count"
     assert len(y_chunks) == expected_num_chunks, "Incorrect y chunk count"
@@ -63,25 +60,19 @@ def test_split_large_data():
     np.testing.assert_array_equal(reconstructed_y, large_y, "Reconstructed y differs")
 
     # Test edge case: empty input
-    x_empty, y_empty = split_large_data(
-        [], [], max_chunk, equal_split_size=equal_split_size
-    )
+    x_empty, y_empty = split_large_data([], [], max_chunk)
     assert x_empty == [], "X should be empty list for empty input"
     assert y_empty == [], "y should be empty list for empty input"
 
     # Test edge case: max_data_size >= total_size
-    x_single, y_single = split_large_data(
-        large_x, large_y, total_size + 5, equal_split_size=equal_split_size
-    )
+    x_single, y_single = split_large_data(large_x, large_y, total_size + 5)
     assert len(x_single) == 1, "Should be 1 X chunk if max_size is large"
     assert len(y_single) == 1, "Should be 1 y chunk if max_size is large"
     np.testing.assert_array_equal(x_single[0], large_x)
     np.testing.assert_array_equal(y_single[0], large_y)
 
     # Test edge case: max_data_size = 1
-    x_max_one, y_max_one = split_large_data(
-        large_x, large_y, 1, equal_split_size=equal_split_size
-    )
+    x_max_one, y_max_one = split_large_data(large_x, large_y, 1)
     assert len(x_max_one) == total_size, "Should be total_size chunks if max_size=1"
     assert len(y_max_one) == total_size, "Should be total_size chunks if max_size=1"
     assert len(x_max_one[0]) == 1, "Each X chunk should have size 1"
@@ -92,9 +83,7 @@ def test_split_large_data():
     max_chunk_exact = 30
     large_x_exact = np.arange(total_size_exact * 2).reshape((total_size_exact, 2))
     large_y_exact = np.arange(total_size_exact)
-    x_exact, y_exact = split_large_data(
-        large_x_exact, large_y_exact, max_chunk_exact, equal_split_size=equal_split_size
-    )
+    x_exact, y_exact = split_large_data(large_x_exact, large_y_exact, max_chunk_exact)
     assert len(x_exact) == total_size_exact // max_chunk_exact  # Should be 3 chunks
     assert len(y_exact) == total_size_exact // max_chunk_exact  # Should be 3 chunks
     assert len(x_exact[0]) == max_chunk_exact  # All chunks should be max size
diff --git a/tests/test_inference.py b/tests/test_inference.py
deleted file mode 100644
index 597de40..0000000
--- a/tests/test_inference.py
+++ /dev/null
@@ -1,217 +0,0 @@
-"""Test the inference engines."""
-
-from __future__ import annotations
-
-from typing import Literal, overload
-from typing_extensions import override
-
-import pytest
-import torch
-from numpy.random import default_rng
-from torch import Tensor
-
-from tabpfn.architectures.interface import Architecture
-from tabpfn.inference import InferenceEngineCachePreprocessing, InferenceEngineOnDemand
-from tabpfn.preprocessing import (
-    ClassifierEnsembleConfig,
-    EnsembleConfig,
-    PreprocessorConfig,
-)
-
-
-class TestModel(Architecture):
-    @overload
-    def forward(
-        self,
-        x: Tensor | dict[str, Tensor],
-        y: Tensor | dict[str, Tensor] | None,
-        *,
-        only_return_standard_out: Literal[True] = True,
-        categorical_inds: list[list[int]] | None = None,
-    ) -> Tensor: ...
-
-    @overload
-    def forward(
-        self,
-        x: Tensor | dict[str, Tensor],
-        y: Tensor | dict[str, Tensor] | None,
-        *,
-        only_return_standard_out: Literal[False],
-        categorical_inds: list[list[int]] | None = None,
-    ) -> dict[str, Tensor]: ...
-
-    @override
-    def forward(
-        self,
-        x: Tensor | dict[str, Tensor],
-        y: Tensor | dict[str, Tensor] | None,
-        *,
-        only_return_standard_out: bool = True,
-        categorical_inds: list[list[int]] | None = None,
-    ) -> Tensor | dict[str, Tensor]:
-        """Perform a forward pass, see doc string of `Architecture`."""
-        assert isinstance(x, Tensor)
-        assert isinstance(y, Tensor)
-        n_train_test, _, _ = x.shape
-        n_train, _ = y.shape
-        test_rows = n_train_test - n_train
-        return x.sum(-2, keepdim=True).sum(-1, keepdim=True).reshape(-1, test_rows)
-
-    @property
-    def ninp(self) -> int:
-        return 2
-
-    @property
-    def features_per_group(self) -> int:
-        return 2
-
-    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
-        pass
-
-
-def test__cache_preprocessing__result_equal_in_serial_and_in_parallel() -> None:
-    rng = default_rng(seed=0)
-    n_train = 100
-    n_features = 4
-    n_classes = 3
-    X_train = rng.standard_normal(size=(n_train, n_features))
-    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
-    X_test = rng.standard_normal(size=(2, n_features))
-
-    engine = InferenceEngineCachePreprocessing.prepare(
-        X_train,
-        y_train,
-        cat_ix=[] * n_train,
-        models=[TestModel()],
-        ensemble_configs=_create_test_ensemble_configs(
-            n_configs=5,
-            n_classes=n_classes,
-            num_models=1,
-        ),
-        # We want to test n_preprocessing_jobs>1 as this might mean the outputs are not
-        # in the same order as the input configs, and we want to check that the parallel
-        # evaluation code behaves correctly in this scenario.
-        n_preprocessing_jobs=5,
-        rng=rng,
-        dtype_byte_size=4,
-        force_inference_dtype=None,
-        save_peak_mem=True,
-        inference_mode=True,
-    )
-
-    outputs_sequential = list(
-        engine.iter_outputs(X_test, devices=[torch.device("cpu")], autocast=False)
-    )
-    outputs_parallel = list(
-        engine.iter_outputs(
-            X_test, devices=[torch.device("cpu"), torch.device("cpu")], autocast=False
-        )
-    )
-
-    assert len(outputs_sequential) == len(outputs_parallel)
-    for par_output, par_config in outputs_parallel:
-        seq_output = _find_seq_output(par_config, outputs_sequential)
-        assert isinstance(seq_output, Tensor)
-        assert isinstance(par_output, Tensor)
-        assert torch.allclose(seq_output, par_output)
-
-
-def test__on_demand__result_equal_in_serial_and_in_parallel() -> None:
-    rng = default_rng(seed=0)
-    n_train = 100
-    n_features = 4
-    n_classes = 3
-    X_train = rng.standard_normal(size=(n_train, n_features))
-    y_train = rng.integers(low=0, high=n_classes - 1, size=(n_train, 1))
-    X_test = rng.standard_normal(size=(2, n_features))
-
-    num_models = 3
-    models = [TestModel() for _ in range(num_models)]
-    engine = InferenceEngineOnDemand.prepare(
-        X_train,
-        y_train,
-        cat_ix=[] * n_train,
-        models=models,
-        ensemble_configs=_create_test_ensemble_configs(
-            n_configs=5,
-            n_classes=3,
-            num_models=num_models,
-        ),
-        # We want to test n_preprocessing_jobs>1 as this might mean the outputs are not
-        # in the same order as the input configs, and we want to check that the parallel
-        # evaluation code behaves correctly in this scenario.
-        n_preprocessing_jobs=5,
-        rng=rng,
-        dtype_byte_size=4,
-        force_inference_dtype=None,
-        save_peak_mem=True,
-    )
-
-    outputs_sequential = list(
-        engine.iter_outputs(X_test, devices=[torch.device("cpu")], autocast=False)
-    )
-    outputs_parallel = list(
-        engine.iter_outputs(
-            X_test, devices=[torch.device("cpu"), torch.device("cpu")], autocast=False
-        )
-    )
-
-    assert len(outputs_sequential) == len(outputs_parallel)
-    last_model_index = 0
-    for par_output, par_config in outputs_parallel:
-        # Test that models are executed in order.
-        assert par_config._model_index >= last_model_index
-        seq_output = _find_seq_output(par_config, outputs_sequential)
-        assert isinstance(seq_output, Tensor)
-        assert isinstance(par_output, Tensor)
-        assert torch.allclose(seq_output, par_output)
-        last_model_index = par_config._model_index
-
-
-def _create_test_ensemble_configs(
-    n_configs: int,
-    n_classes: int,
-    num_models: int,
-) -> list[ClassifierEnsembleConfig]:
-    preprocessor_configs = [
-        PreprocessorConfig(
-            "quantile_uni_coarse",
-            append_original="auto",
-            categorical_name="ordinal_very_common_categories_shuffled",
-            global_transformer_name="svd",
-            max_features_per_estimator=500,
-        ),
-        PreprocessorConfig(
-            "none",
-            categorical_name="numeric",
-            max_features_per_estimator=500,
-        ),
-    ]
-    return EnsembleConfig.generate_for_classification(
-        num_estimators=n_configs,
-        subsample_size=None,
-        max_index=n_classes - 1,
-        add_fingerprint_feature=True,
-        polynomial_features="all",
-        feature_shift_decoder="shuffle",
-        preprocessor_configs=preprocessor_configs,
-        class_shift_method=None,
-        n_classes=n_classes,
-        random_state=0,
-        num_models=num_models,
-    )
-
-
-def _find_seq_output(
-    config: EnsembleConfig,
-    outputs_sequential: list[tuple[Tensor | dict, EnsembleConfig]],
-) -> Tensor | dict:
-    """Find the sequential output corresponding to the given config.
-
-    The configs are not hashable, so we have to resort to this search method.
-    """
-    for output, trial_config in outputs_sequential:
-        if trial_config == config:
-            return output
-
-    return pytest.fail(f"Parallel config was not found in sequential configs: {config}")
diff --git a/tests/test_inference_config.py b/tests/test_inference_config.py
deleted file mode 100644
index d55d481..0000000
--- a/tests/test_inference_config.py
+++ /dev/null
@@ -1,67 +0,0 @@
-"""Tests for the InferenceConfig."""
-
-from __future__ import annotations
-
-import io
-from dataclasses import asdict
-
-import torch
-
-from tabpfn.constants import ModelVersion
-from tabpfn.inference_config import InferenceConfig
-from tabpfn.preprocessing import PreprocessorConfig
-
-
-def test__save_and_load__loaded_value_equal_to_saved() -> None:
-    config = InferenceConfig.get_default(task_type="multiclass", model_version="latest")
-
-    with io.BytesIO() as buffer:
-        torch.save(asdict(config), buffer)
-        buffer.seek(0)
-        loaded_config = InferenceConfig(**torch.load(buffer, weights_only=False))
-
-    assert loaded_config == config
-
-
-def test__override_with_user_input__dict_of_overrides__sets_values_correctly() -> None:
-    config = InferenceConfig.get_default(
-        task_type="multiclass", model_version=ModelVersion.V2
-    )
-    overrides = {
-        "PREPROCESS_TRANSFORMS": [
-            {
-                "name": "adaptive",
-                "append_original": "auto",
-                "categorical_name": "ordinal_very_common_categories_shuffled",
-                "global_transformer_name": "svd",
-            }
-        ],
-        "POLYNOMIAL_FEATURES": "all",
-    }
-    new_config = config.override_with_user_input(overrides)
-    assert new_config is not config
-    assert new_config != config
-    assert isinstance(new_config.PREPROCESS_TRANSFORMS[0], PreprocessorConfig)
-    assert new_config.PREPROCESS_TRANSFORMS[0].name == "adaptive"
-    assert new_config.POLYNOMIAL_FEATURES == "all"
-
-
-def test__override_with_user_input__config_override__replaces_entire_config() -> None:
-    config = InferenceConfig.get_default(
-        task_type="regression", model_version=ModelVersion.V2
-    )
-    override_config = InferenceConfig(
-        PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="adaptive")],
-        POLYNOMIAL_FEATURES="all",
-    )
-    new_config = config.override_with_user_input(override_config)
-    assert new_config is not config
-    assert new_config != config
-    assert new_config == override_config
-
-
-def test__override_with_user_input__override_is_None__returns_copy_of_config() -> None:
-    config = InferenceConfig.get_default(task_type="regression", model_version="latest")
-    new_config = config.override_with_user_input(user_config=None)
-    assert new_config is not config
-    assert new_config == config
diff --git a/tests/test_inference_tuning.py b/tests/test_inference_tuning.py
deleted file mode 100644
index cb71025..0000000
--- a/tests/test_inference_tuning.py
+++ /dev/null
@@ -1,195 +0,0 @@
-from __future__ import annotations
-
-import numpy as np
-import pytest
-
-from tabpfn.inference_tuning import (
-    ClassifierEvalMetrics,
-    ClassifierTuningConfig,
-    find_optimal_classification_threshold_single_class,
-    find_optimal_classification_thresholds,
-    resolve_tuning_config,
-    select_robust_optimal_threshold,
-)
-
-
-@pytest.mark.parametrize(
-    ("y_true", "y_pred_probs", "expected_interval"),
-    [
-        (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), (0.3, 0.7)),
-        (np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6]), (0.3, 0.7)),
-    ],
-)
-def test__find_optimal_classification_threshold_single_class__threshold_in_interval(
-    y_true: np.ndarray,
-    y_pred_probs: np.ndarray,
-    expected_interval: tuple[float, float],
-) -> None:
-    best_threshold = find_optimal_classification_threshold_single_class(
-        metric_name=ClassifierEvalMetrics.F1,
-        y_true=y_true,
-        y_pred_probas=y_pred_probs,
-    )
-    lo, hi = expected_interval
-    assert lo <= best_threshold <= hi
-
-
-@pytest.mark.parametrize(
-    ("thresholds_and_losses", "expected_threshold", "plateau_delta"),
-    [
-        ([(1, 0.4), (2, 0.3), (3, 0.301), (4, 0.3015), (5, 0.6)], 3.0, 0.0018),
-        ([(1, 0.2), (2, 0.1), (3, 0.101), (4, 0.1015), (5, 0.05)], 5.0, 0.002),
-        ([(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.2)], 3.0, 0.2),
-        ([(1, 0.1), (2, 0.5), (3, 0.6), (4, 0.7), (5, 0.8)], 1.0, 0.001),
-        ([(1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.1)], 5.0, 0.001),
-        ([(1, 0.3), (2, 0.1), (3, 0.11), (4, 0.5)], 2.0, 0.005),
-        ([(1, 0.2), (2, 0.2), (3, 0.6), (4, 0.7)], 2.0, 0.0001),
-        ([(1, 0.1), (2, 0.11), (3, 0.12), (4, 0.11), (5, 0.1)], 1.0, 0.002),
-        ([(1, 0.1), (2, 0.11), (3, 0.12), (4, 0.11), (5, 0.1)], 3.0, 0.5),
-        ([(1, 0.3), (2, 0.2), (3, 0.21), (4, 0.22), (5, 0.23)], 2.0, 0.01),
-        ([(1, 0.5), (2, 0.3), (3, 0.2), (4, 0.21), (5, 0.21)], 4.0, 0.01),
-        ([(1, 0.4), (2, 0.4), (3, 0.1), (4, 0.4), (5, 0.4)], 3.0, 0.001),
-        ([(1, 0.1), (2, 0.101), (3, 0.102)], 2.0, 0.002),
-    ],
-)
-def test__select_robust_optimal_threshold__works_as_expected(
-    thresholds_and_losses: list[tuple[float, float]],
-    expected_threshold: float,
-    plateau_delta: float,
-) -> None:
-    assert (
-        select_robust_optimal_threshold(
-            thresholds_and_losses=thresholds_and_losses,
-            plateau_delta=plateau_delta,
-        )
-        == expected_threshold
-    )
-
-
-@pytest.mark.parametrize(
-    (
-        "y_true",
-        "y_pred_probas",
-        "expected_thresholds",
-    ),
-    [
-        (
-            np.array([0, 1, 2, 0, 1, 2]),
-            np.array(
-                [
-                    [0.9, 0.05, 0.05],
-                    [0.05, 0.9, 0.05],
-                    [0.05, 0.05, 0.9],
-                    [0.9, 0.05, 0.05],
-                    [0.05, 0.9, 0.05],
-                    [0.05, 0.05, 0.9],
-                ]
-            ),
-            [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95)],
-        ),
-        (
-            np.array([0, 0, 0, 1, 1, 1, 2, 2]),
-            np.array(
-                [
-                    [0.8, 0.1, 0.1],
-                    [0.85, 0.08, 0.07],
-                    [0.75, 0.15, 0.1],
-                    [0.15, 0.7, 0.15],
-                    [0.1, 0.8, 0.1],
-                    [0.2, 0.75, 0.05],
-                    [0.1, 0.1, 0.8],
-                    [0.05, 0.15, 0.8],
-                ]
-            ),
-            [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95)],
-        ),
-        (
-            np.array([0, 0, 1, 1, 2, 2]),
-            np.array(
-                [
-                    [0.9, 0.05, 0.05],
-                    [0.70, 0.25, 0.05],
-                    [0.3, 0.6, 0.1],
-                    [0.25, 0.65, 0.1],
-                    [0.2, 0.1, 0.7],
-                    [0.15, 0.15, 0.7],
-                ]
-            ),
-            [(0.45, 0.95), (0.4, 0.75), (0.3, 0.8)],
-        ),
-        (
-            np.array([0, 0, 0, 1, 1, 2]),
-            np.array(
-                [
-                    [0.95, 0.03, 0.02],
-                    [0.9, 0.05, 0.05],
-                    [0.88, 0.07, 0.05],
-                    [0.4, 0.55, 0.05],
-                    [0.35, 0.6, 0.05],
-                    [0.1, 0.1, 0.8],
-                ]
-            ),
-            [(0.6, 0.95), (0.1, 0.5), (0.05, 0.95)],
-        ),
-    ],
-)
-def test__find_optimal_classification_thresholds__works_for_multiclass_f1(
-    y_true: np.ndarray,
-    y_pred_probas: np.ndarray,
-    expected_thresholds: list[tuple[float, float]],
-) -> None:
-    thresholds = find_optimal_classification_thresholds(
-        metric_name=ClassifierEvalMetrics.F1,
-        y_true=y_true,
-        y_pred_probas=y_pred_probas,
-        n_classes=len(expected_thresholds),
-    )
-
-    assert thresholds.shape == (len(expected_thresholds),)
-    for i, (lo, hi) in enumerate(expected_thresholds):
-        assert lo <= thresholds[i] <= hi, (
-            f"Threshold for class {i} is {thresholds[i]}, "
-            f"expected to be in [{lo}, {hi}]"
-        )
-
-
-@pytest.mark.parametrize(
-    (
-        "X_train_shape",
-        "tune_decision_thresholds",
-        "calibrate_temperature",
-        "expected_tuning_holdout_pct",
-        "expected_tuning_holdout_n_splits",
-    ),
-    [
-        ((1_000, 10), False, True, 0.1, 10),
-        ((9_000, 10), False, True, 0.2, 3),
-        ((9_000, 10), True, False, 0.2, 3),
-        ((20_000, 10), True, False, 0.2, 2),
-        ((21_000, 10), True, False, 0.3, 1),
-    ],
-)
-def test__resolve_tuning_config__provides_expected_values_for_auto_config(
-    X_train_shape: tuple[int, int],
-    calibrate_temperature: bool,
-    tune_decision_thresholds: bool,
-    expected_tuning_holdout_pct: float,
-    expected_tuning_holdout_n_splits: int,
-) -> None:
-    tuning_config = ClassifierTuningConfig(
-        calibrate_temperature=calibrate_temperature,
-        tune_decision_thresholds=tune_decision_thresholds,
-        tuning_holdout_frac="auto",
-        tuning_n_folds="auto",
-    )
-    resolved_tuning_config = resolve_tuning_config(
-        tuning_config=tuning_config,
-        num_samples=X_train_shape[0],
-    )
-    assert isinstance(resolved_tuning_config, ClassifierTuningConfig)
-
-    assert resolved_tuning_config is not None
-    assert resolved_tuning_config.calibrate_temperature == calibrate_temperature
-    assert resolved_tuning_config.tune_decision_thresholds == tune_decision_thresholds
-    assert resolved_tuning_config.tuning_holdout_frac == expected_tuning_holdout_pct
-    assert resolved_tuning_config.tuning_n_folds == expected_tuning_holdout_n_splits
diff --git a/tests/test_model/test_attention.py b/tests/test_model/test_attention.py
index ce645dc..d4350d9 100644
--- a/tests/test_model/test_attention.py
+++ b/tests/test_model/test_attention.py
@@ -1,8 +1,6 @@
 from __future__ import annotations
 
-import pytest
 import torch
-from torch.nn.functional import scaled_dot_product_attention
 
 from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
 from tabpfn.architectures.base.config import ModelConfig
@@ -112,33 +110,5 @@ def test_attention():
     assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5
 
 
-@pytest.mark.parametrize(
-    ("batch_size", "seq_len", "num_heads", "head_dim"),
-    [
-        (100, 64, 8, 32),
-        (1100, 16, 2, 8),  # Large batch (will be chunked)
-    ],
-)
-def test_scaled_dot_product_attention_chunked(
-    batch_size: int, seq_len: int, num_heads: int, head_dim: int
-) -> None:
-    """Test that scaled_dot_product_attention_chunked is
-    equivalent to torch scaled_dot_product_attention.
-    """
-    torch.manual_seed(42)
-    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
-    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
-    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
-
-    # Test with dropout disabled for deterministic comparison
-    dropout_p = 0.0
-
-    torch.manual_seed(42)
-    original_output = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
-
-    torch.manual_seed(42)
-    chunked_output = MultiHeadAttention.scaled_dot_product_attention_chunked(
-        q, k, v, dropout_p=dropout_p, max_batch_size=500
-    )
-
-    torch.testing.assert_close(original_output, chunked_output, rtol=1e-5, atol=1e-6)
+if __name__ == "__main__":
+    test_attention()
diff --git a/tests/test_model/test_encoders.py b/tests/test_model/test_encoders.py
index 58f7919..535be02 100644
--- a/tests/test_model/test_encoders.py
+++ b/tests/test_model/test_encoders.py
@@ -10,7 +10,6 @@ from tabpfn.architectures.base import encoders
 from tabpfn.architectures.base.encoders import (
     InputNormalizationEncoderStep,
     LinearInputEncoderStep,
-    MLPInputEncoderStep,
     NanHandlingEncoderStep,
     RemoveEmptyFeaturesEncoderStep,
     SequentialEncoder,
@@ -77,13 +76,9 @@ def test_normalize_data_basic(dtype, shape):
     mean_of_norm = x_norm.mean(dim=0)
     std_of_norm = x_norm.std(dim=0)
 
-    # For dtype torch.float16 1e-3 is too much precision and results in
-    # randomly failing tests, due to precision. Therefore increase the
-    # tolerance.
-    atol = 1e-2 if dtype == torch.float16 else 1e-3
     # Assert that mean is close to 0 and std is close to 1 for each feature
-    assert torch.allclose(mean_of_norm, torch.zeros_like(mean_of_norm), atol=atol)
-    assert torch.allclose(std_of_norm, torch.ones_like(std_of_norm), atol=atol)
+    assert torch.allclose(mean_of_norm, torch.zeros_like(mean_of_norm), atol=1e-3)
+    assert torch.allclose(std_of_norm, torch.ones_like(std_of_norm), atol=1e-3)
     assert not torch.isnan(x_norm).any()
     assert not torch.isinf(x_norm).any()
 
@@ -123,13 +118,13 @@ def test_input_normalization():
     )
 
     out = encoder({"main": x}, single_eval_pos=-1)["main"]
-    assert torch.isclose(out.var(dim=0), torch.tensor([1.0]), atol=1e-05).all(), (
-        "Variance should be 1.0 for all features and batch samples."
-    )
+    assert torch.isclose(
+        out.var(dim=0), torch.tensor([1.0]), atol=1e-05
+    ).all(), "Variance should be 1.0 for all features and batch samples."
 
-    assert torch.isclose(out.mean(dim=0), torch.tensor([0.0]), atol=1e-05).all(), (
-        "Mean should be 0.0 for all features and batch samples."
-    )
+    assert torch.isclose(
+        out.mean(dim=0), torch.tensor([0.0]), atol=1e-05
+    ).all(), "Mean should be 0.0 for all features and batch samples."
 
     out = encoder({"main": x}, single_eval_pos=5)["main"]
     assert torch.isclose(out[0:5].var(dim=0), torch.tensor([1.0]), atol=1e-03).all(), (
@@ -146,12 +141,12 @@ def test_input_normalization():
     x[:, 1, :] = 100.0
     x[:, 2, 6:] = 100.0
     out = encoder({"main": x}, single_eval_pos=5)["main"]
-    assert (out[:, 0, :] == out_ref[:, 0, :]).all(), (
-        "Changing one batch should not affeect the others."
-    )
-    assert (out[:, 2, 0:5] == out_ref[:, 2, 0:5]).all(), (
-        "Changing unnormalized part of the batch should not affect the others."
-    )
+    assert (
+        out[:, 0, :] == out_ref[:, 0, :]
+    ).all(), "Changing one batch should not affeect the others."
+    assert (
+        out[:, 2, 0:5] == out_ref[:, 2, 0:5]
+    ).all(), "Changing unnormalized part of the batch should not affect the others."
 
 
 def test_remove_empty_feats():
@@ -169,21 +164,21 @@ def test_remove_empty_feats():
 
     x[0, 1, 1] = 0.0
     out = encoder({"main": x}, single_eval_pos=-1)["main"]
-    assert (out[:, 1, -1] != 0).all(), (
-        "Should not change anything if no column is entirely empty."
-    )
+    assert (
+        out[:, 1, -1] != 0
+    ).all(), "Should not change anything if no column is entirely empty."
 
     x[:, 1, 1] = 0.0
     out = encoder({"main": x}, single_eval_pos=-1)["main"]
-    assert (out[:, 1, -1] == 0).all(), (
-        "Empty column should be removed and shifted to the end."
-    )
-    assert (out[:, 1, 1] != 0).all(), (
-        "The place of the empty column should be filled with the next column."
-    )
-    assert (out[:, 2, 1] != 0).all(), (
-        "Non empty columns should not be changed in their position."
-    )
+    assert (
+        out[:, 1, -1] == 0
+    ).all(), "Empty column should be removed and shifted to the end."
+    assert (
+        out[:, 1, 1] != 0
+    ).all(), "The place of the empty column should be filled with the next column."
+    assert (
+        out[:, 2, 1] != 0
+    ).all(), "Non empty columns should not be changed in their position."
 
 
 def test_variable_num_features():
@@ -197,9 +192,9 @@ def test_variable_num_features():
     )
 
     out = encoder({"main": x}, single_eval_pos=-1)["main"]
-    assert out.shape[-1] == fixed_out, (
-        "Features were not extended to the requested number of features."
-    )
+    assert (
+        out.shape[-1] == fixed_out
+    ), "Features were not extended to the requested number of features."
     assert torch.isclose(
         out[:, :, 0 : x.shape[-1]] / x, torch.tensor([math.sqrt(fixed_out / F)])
     ).all(), "Normalization is not correct."
@@ -218,9 +213,9 @@ def test_variable_num_features():
         VariableNumFeaturesEncoderStep(**kwargs), output_key=None
     )
     out = encoder({"main": x}, single_eval_pos=-1)["main"]
-    assert (out[:, :, : x.shape[-1]] == x).all(), (
-        "Features should be unchanged when not normalizing."
-    )
+    assert (
+        out[:, :, : x.shape[-1]] == x
+    ).all(), "Features should be unchanged when not normalizing."
 
 
 def test_nan_handling_encoder():
@@ -266,48 +261,6 @@ def test_linear_encoder():
     assert out.shape[-1] == F, "Output should have the requested number of features."
 
 
-@pytest.mark.parametrize("num_layers", [2, 3])
-def test__MLPInputEncoderStep__embed_each_input_cell(num_layers):
-    """Test MLP encoder input/output dimensions."""
-    N, B, F = 10, 3, 4
-    emsize = 8
-    x = torch.randn([N, B, F])
-
-    # Test basic MLP encoder with default hidden_dim (should equal emsize)
-    encoder = SequentialEncoder(
-        MLPInputEncoderStep(
-            num_features=F,
-            emsize=emsize,
-            num_layers=num_layers,
-        ),
-        output_key=None,
-    )
-    out = encoder({"main": x}, single_eval_pos=-1)["output"]
-    assert out.shape == (
-        N,
-        B,
-        emsize,
-    ), f"Output shape should be ({N}, {B}, {emsize}), got {out.shape}"
-
-    # Test with explicit hidden_dim
-    hidden_dim = 16
-    encoder = SequentialEncoder(
-        MLPInputEncoderStep(
-            num_features=F,
-            emsize=emsize,
-            hidden_dim=hidden_dim,
-            num_layers=num_layers,
-        ),
-        output_key=None,
-    )
-    out = encoder({"main": x}, single_eval_pos=-1)["output"]
-    assert out.shape == (
-        N,
-        B,
-        emsize,
-    ), f"Output shape should be ({N}, {B}, {emsize}), got {out.shape}"
-
-
 def test_combination():
     N, B, F, fixed_out = 10, 3, 4, 5
     x = torch.randn([N, B, F])
@@ -348,12 +301,12 @@ def test_combination():
     x[:, 1, :] = 100.0
     x[6:, 2, 2] = 100.0
     out = encoder({"main": x}, single_eval_pos=5)["main"]
-    assert (out[:, 0, :] == out_ref[:, 0, :]).all(), (
-        "Changing one batch should not affeect the others."
-    )
-    assert (out[0:5, 2, 2] == out_ref[0:5, 2, 2]).all(), (
-        "Changing unnormalized part of the batch should not affect the others."
-    )
+    assert (
+        out[:, 0, :] == out_ref[:, 0, :]
+    ).all(), "Changing one batch should not affeect the others."
+    assert (
+        out[0:5, 2, 2] == out_ref[0:5, 2, 2]
+    ).all(), "Changing unnormalized part of the batch should not affect the others."
 
     x = torch.randn([N, B, F])
     x[1, 0, 2] = np.inf
@@ -379,12 +332,12 @@ def test_combination():
         {"main": x_param, "domain_indicator": domain_indicator}, single_eval_pos=5
     )["main"].sum()
     s.backward()
-    assert x_param.grad is not None, (
-        "the encoder is not differentiable, i.e. the gradients are None."
-    )
-    assert not torch.isnan(x_param.grad).any(), (
-        "the encoder is not differentiable, i.e. the gradients are nan."
-    )
+    assert (
+        x_param.grad is not None
+    ), "the encoder is not differentiable, i.e. the gradients are None."
+    assert not torch.isnan(
+        x_param.grad
+    ).any(), "the encoder is not differentiable, i.e. the gradients are nan."
 
 
 def test_multiclass_encoder():
@@ -409,10 +362,7 @@ def test_interface():
             and cls is not encoders.SeqEncStep
         ):
             num_features = 4
-            if (
-                cls is encoders.LinearInputEncoderStep
-                or cls is encoders.MLPInputEncoderStep
-            ):
+            if cls is encoders.LinearInputEncoderStep:
                 obj = cls(num_features=num_features, emsize=16)
             elif cls is encoders.VariableNumFeaturesEncoderStep:
                 obj = cls(num_features=num_features)
diff --git a/tests/test_model/test_seperate_train_inference.py b/tests/test_model/test_seperate_train_inference.py
index 59a5b6d..030ac65 100644
--- a/tests/test_model/test_seperate_train_inference.py
+++ b/tests/test_model/test_seperate_train_inference.py
@@ -82,6 +82,6 @@ def test_separate_train_inference(multiquery_item_attention_for_test_set: bool):
     torch.manual_seed(12345)
     logits1a = model(x=torch.concat([x_train, x_test]), y=y)
 
-    assert logits1.float() == pytest.approx(logits1a.float(), abs=1e-5), (
-        f"{logits1} != {logits1a}"
-    )
+    assert logits1.float() == pytest.approx(
+        logits1a.float(), abs=1e-5
+    ), f"{logits1} != {logits1a}"
diff --git a/tests/test_model_loading.py b/tests/test_model_loading.py
index f40d5dd..c22e58c 100644
--- a/tests/test_model_loading.py
+++ b/tests/test_model_loading.py
@@ -6,7 +6,6 @@ from typing import Any, Literal, overload
 from typing_extensions import override
 from unittest.mock import patch
 
-import pytest
 import torch
 from pydantic.dataclasses import dataclass
 from torch import Tensor
@@ -20,8 +19,6 @@ from tabpfn.architectures.interface import (
     ArchitectureConfig,
     ArchitectureModule,
 )
-from tabpfn.inference_config import InferenceConfig
-from tabpfn.preprocessing import PreprocessorConfig
 
 
 def test__load_model__no_architecture_name_in_checkpoint__loads_base_architecture(
@@ -33,7 +30,7 @@ def test__load_model__no_architecture_name_in_checkpoint__loads_base_architectur
     checkpoint_path = tmp_path / "checkpoint.ckpt"
     torch.save(checkpoint, checkpoint_path)
 
-    loaded_model, _, loaded_config, _ = model_loading.load_model(path=checkpoint_path)
+    loaded_model, _, loaded_config = model_loading.load_model(path=checkpoint_path)
     assert isinstance(loaded_model, PerFeatureTransformer)
     assert isinstance(loaded_config, ModelConfig)
 
@@ -128,231 +125,6 @@ def test__load_model__architecture_name_in_checkpoint__loads_specified_architect
     checkpoint_path = tmp_path / "checkpoint.ckpt"
     torch.save(checkpoint, checkpoint_path)
 
-    loaded_model, _, loaded_config, _ = model_loading.load_model(path=checkpoint_path)
+    loaded_model, _, loaded_config = model_loading.load_model(path=checkpoint_path)
     assert isinstance(loaded_model, DummyArchitecture)
     assert isinstance(loaded_config, FakeConfig)
-
-
-def test__load_v2_checkpoint__returns_v2_preprocessings(
-    tmp_path: Path,
-) -> None:
-    architecture_config = _get_minimal_base_architecture_config()
-    model = base.get_architecture(
-        architecture_config, n_out=10, cache_trainset_representation=True
-    )
-    # v2 checkpoints have no "architecture_name" key
-    checkpoint = {
-        "state_dict": model.state_dict(),
-        "config": asdict(architecture_config),
-    }
-    checkpoint_path = tmp_path / "checkpoint.ckpt"
-    torch.save(checkpoint, checkpoint_path)
-
-    _, _, _, inference_config = model_loading.load_model_criterion_config(
-        model_path=[checkpoint_path, checkpoint_path],
-        check_bar_distribution_criterion=False,
-        cache_trainset_representation=False,
-        which="classifier",
-        version="v2",
-        download_if_not_exists=False,
-    )
-
-    assert len(inference_config.PREPROCESS_TRANSFORMS) == 2
-    assert inference_config.PREPROCESS_TRANSFORMS[0].name == "quantile_uni_coarse"
-    assert inference_config.PREPROCESS_TRANSFORMS[0].append_original == "auto"
-    assert (
-        inference_config.PREPROCESS_TRANSFORMS[0].categorical_name
-        == "ordinal_very_common_categories_shuffled"
-    )
-    assert inference_config.PREPROCESS_TRANSFORMS[0].global_transformer_name == "svd"
-    assert (
-        inference_config.PREPROCESS_TRANSFORMS[0].max_features_per_estimator
-        == 1_000_000
-    )
-    assert inference_config.PREPROCESS_TRANSFORMS[1].name == "none"
-    assert inference_config.PREPROCESS_TRANSFORMS[1].categorical_name == "numeric"
-    assert (
-        inference_config.PREPROCESS_TRANSFORMS[1].max_features_per_estimator
-        == 1_000_000
-    )
-
-
-@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
-def test__load_v2_5_classification_ckpt__returns_v2_5_preprocessing(
-    tmp_path: Path,
-) -> None:
-    # v2.5 checkpoints have a architecture_name but no inference_config
-    # classification checkpoints have max_num_classes > 0
-    architecture_config = {"max_num_classes": 10, "num_buckets": 100}
-    checkpoint = {
-        "state_dict": {},
-        "config": architecture_config,
-        "architecture_name": "fake_arch",
-    }
-    checkpoint_path = tmp_path / "checkpoint.ckpt"
-    torch.save(checkpoint, checkpoint_path)
-
-    _, _, _, inference_config = model_loading.load_model_criterion_config(
-        model_path=[checkpoint_path, checkpoint_path],
-        check_bar_distribution_criterion=False,
-        cache_trainset_representation=False,
-        which="classifier",
-        version="v2.5",
-        download_if_not_exists=False,
-    )
-
-    assert len(inference_config.PREPROCESS_TRANSFORMS) == 2
-    assert inference_config.PREPROCESS_TRANSFORMS[0].name == "squashing_scaler_default"
-    assert inference_config.PREPROCESS_TRANSFORMS[0].append_original is False
-    assert (
-        inference_config.PREPROCESS_TRANSFORMS[0].categorical_name
-        == "ordinal_very_common_categories_shuffled"
-    )
-    assert (
-        inference_config.PREPROCESS_TRANSFORMS[0].global_transformer_name
-        == "svd_quarter_components"
-    )
-    assert inference_config.PREPROCESS_TRANSFORMS[0].max_features_per_estimator == 500
-    assert inference_config.PREPROCESS_TRANSFORMS[1].name == "none"
-    assert inference_config.PREPROCESS_TRANSFORMS[1].categorical_name == "numeric"
-    assert inference_config.PREPROCESS_TRANSFORMS[1].max_features_per_estimator == 500
-
-
-@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
-def test__load_v2_5_regression_ckpt__returns_v2_5_preprocessing(
-    tmp_path: Path,
-) -> None:
-    # v2.5 checkpoints have a architecture_name but no inference_config
-    # regression checkpoints have max_num_classes 0
-    architecture_config = {"max_num_classes": 0, "num_buckets": 100}
-    checkpoint = {
-        "state_dict": {
-            "criterion.borders": torch.arange(101),
-            "criterion.losses_per_bucket": torch.randn((100,)),
-        },
-        "config": architecture_config,
-        "architecture_name": "fake_arch",
-    }
-    checkpoint_path = tmp_path / "checkpoint.ckpt"
-    torch.save(checkpoint, checkpoint_path)
-
-    _, _, _, inference_config = model_loading.load_model_criterion_config(
-        model_path=[checkpoint_path, checkpoint_path],
-        check_bar_distribution_criterion=False,
-        cache_trainset_representation=False,
-        which="classifier",
-        version="v2.5",
-        download_if_not_exists=False,
-    )
-
-    assert len(inference_config.PREPROCESS_TRANSFORMS) == 2
-    assert inference_config.PREPROCESS_TRANSFORMS[0].name == "quantile_uni_coarse"
-    assert inference_config.PREPROCESS_TRANSFORMS[0].append_original == "auto"
-    assert inference_config.PREPROCESS_TRANSFORMS[0].categorical_name == "numeric"
-    assert inference_config.PREPROCESS_TRANSFORMS[0].global_transformer_name is None
-    assert inference_config.PREPROCESS_TRANSFORMS[1].name == "squashing_scaler_default"
-    assert (
-        inference_config.PREPROCESS_TRANSFORMS[1].categorical_name
-        == "ordinal_very_common_categories_shuffled"
-    )
-
-
-@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
-def test__load_checkpoints_with_inference_configs__returns_inference_config(
-    tmp_path: Path,
-) -> None:
-    architecture_config = {"max_num_classes": 10, "num_buckets": 100}
-    inference_config = InferenceConfig(
-        PREPROCESS_TRANSFORMS=[
-            PreprocessorConfig(
-                "quantile_uni_coarse",
-                append_original="auto",
-                categorical_name="ordinal_very_common_categories_shuffled",
-                global_transformer_name="svd",
-                max_features_per_estimator=-1,
-            )
-        ]
-    )
-
-    checkpoint_1 = {
-        "state_dict": {},
-        "config": architecture_config,
-        "architecture_name": "fake_arch",
-        "inference_config": asdict(inference_config),
-    }
-    checkpoint_1_path = tmp_path / "checkpoint1.ckpt"
-    torch.save(checkpoint_1, checkpoint_1_path)
-    checkpoint_2 = {
-        "state_dict": {},
-        "config": architecture_config,
-        "architecture_name": "fake_arch",
-        "inference_config": asdict(inference_config),
-    }
-    checkpoint_2_path = tmp_path / "checkpoint2.ckpt"
-    torch.save(checkpoint_2, checkpoint_2_path)
-
-    loaded_models, _, _, loaded_config = model_loading.load_model_criterion_config(
-        model_path=[checkpoint_1_path, checkpoint_2_path],
-        check_bar_distribution_criterion=False,
-        cache_trainset_representation=False,
-        which="classifier",
-        version="v2",
-        download_if_not_exists=False,
-    )
-    assert len(loaded_models) == 2
-    assert loaded_config == inference_config
-
-
-@patch.dict(ARCHITECTURES, fake_arch=FakeArchitectureModule())
-def test__load_multiple_models_with_difference_inference_configs__raises(
-    tmp_path: Path,
-) -> None:
-    architecture_config = {"max_num_classes": 10, "num_buckets": 100}
-    checkpoint_1 = {
-        "state_dict": {},
-        "config": architecture_config,
-        "architecture_name": "fake_arch",
-        "inference_config": asdict(
-            InferenceConfig(
-                PREPROCESS_TRANSFORMS=[
-                    PreprocessorConfig(
-                        "quantile_uni_coarse",
-                        append_original="auto",
-                        categorical_name="ordinal_very_common_categories_shuffled",
-                        global_transformer_name="svd",
-                        max_features_per_estimator=-1,
-                    )
-                ]
-            )
-        ),
-    }
-    checkpoint_1_path = tmp_path / "checkpoint1.ckpt"
-    torch.save(checkpoint_1, checkpoint_1_path)
-    checkpoint_2 = {
-        "state_dict": {},
-        "config": architecture_config,
-        "architecture_name": "fake_arch",
-        "inference_config": asdict(
-            InferenceConfig(
-                PREPROCESS_TRANSFORMS=[
-                    PreprocessorConfig(
-                        "none",
-                        categorical_name="numeric",
-                        max_features_per_estimator=-1,
-                    )
-                ]
-            )
-        ),
-    }
-    checkpoint_2_path = tmp_path / "checkpoint2.ckpt"
-    torch.save(checkpoint_2, checkpoint_2_path)
-
-    with pytest.raises(ValueError, match="Inference configs for different models"):
-        model_loading.load_model_criterion_config(
-            model_path=[checkpoint_1_path, checkpoint_2_path],
-            check_bar_distribution_criterion=False,
-            cache_trainset_representation=False,
-            which="classifier",
-            version="v2",
-            download_if_not_exists=False,
-        )
diff --git a/tests/test_model_move_backwards_compatibility.py b/tests/test_model_move_backwards_compatibility.py
index feb09f4..8395a5b 100644
--- a/tests/test_model_move_backwards_compatibility.py
+++ b/tests/test_model_move_backwards_compatibility.py
@@ -6,22 +6,23 @@ def test__packages_can_still_be_imported_from_old_location() -> None:
 
     We moved the packages from tabpfn.model to tabpfn.architectures.base.
     """
-    import tabpfn.model.attention  # noqa: PLC0415
-    import tabpfn.model.bar_distribution  # noqa: PLC0415
-    import tabpfn.model.config  # noqa: PLC0415
-    import tabpfn.model.encoders  # noqa: PLC0415
-    import tabpfn.model.layer  # noqa: PLC0415
-    import tabpfn.model.loading  # noqa: PLC0415
-    import tabpfn.model.memory  # noqa: PLC0415
-    import tabpfn.model.mlp  # noqa: PLC0415
-    import tabpfn.model.preprocessing  # noqa: PLC0415
-    import tabpfn.model.transformer  # noqa: PLC0415
+    import tabpfn.model.attention
+    import tabpfn.model.bar_distribution
+    import tabpfn.model.config
+    import tabpfn.model.encoders
+    import tabpfn.model.layer
+    import tabpfn.model.loading
+    import tabpfn.model.memory
+    import tabpfn.model.mlp
+    import tabpfn.model.preprocessing
+    import tabpfn.model.transformer
 
     assert hasattr(tabpfn.model.attention, "Attention")
     assert hasattr(tabpfn.model.bar_distribution, "BarDistribution")
     assert hasattr(tabpfn.model.config, "ModelConfig")
     assert hasattr(tabpfn.model.encoders, "InputEncoder")
     assert hasattr(tabpfn.model.layer, "LayerNorm")
+    assert hasattr(tabpfn.model.memory, "MemoryUsageEstimator")
     assert hasattr(tabpfn.model.mlp, "MLP")
     assert hasattr(tabpfn.model.preprocessing, "SequentialFeatureTransformer")
     assert hasattr(tabpfn.model.transformer, "PerFeatureTransformer")
diff --git a/tests/test_parallel_execute.py b/tests/test_parallel_execute.py
deleted file mode 100644
index d3abbf5..0000000
--- a/tests/test_parallel_execute.py
+++ /dev/null
@@ -1,92 +0,0 @@
-"""Tests for tabpfn.parallel_execute."""
-
-from __future__ import annotations
-
-import threading
-
-import torch
-
-from tabpfn.parallel_execute import parallel_execute
-
-
-def test__parallel_execute__single_device__executes_in_current_thread() -> None:
-    def test_function(device: torch.device, is_parallel: bool) -> int:  # noqa: ARG001
-        return threading.get_ident()
-
-    thread_ids = parallel_execute(
-        devices=[torch.device("cpu")], functions=[test_function, test_function]
-    )
-
-    current_thread_id = threading.get_ident()
-    assert list(thread_ids) == [current_thread_id, current_thread_id]
-
-
-def test__parallel_execute__single_device__sets_is_parallel_to_False() -> None:
-    def test_function(device: torch.device, is_parallel: bool) -> bool:  # noqa: ARG001
-        return is_parallel
-
-    is_parallels = parallel_execute(
-        devices=[torch.device("cpu")], functions=[test_function, test_function]
-    )
-
-    assert list(is_parallels) == [False, False]
-
-
-def test__parallel_execute__single_device__results_in_same_order_as_functions() -> None:
-    def a(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
-        return "a"
-
-    def b(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
-        return "b"
-
-    def c(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
-        return "c"
-
-    results = parallel_execute(devices=[torch.device("cpu")], functions=[a, b, c])
-
-    assert list(results) == ["a", "b", "c"]
-
-
-def test__parallel_execute__multiple_devices__executes_in_worker_threads() -> None:
-    def test_function(device: torch.device, is_parallel: bool) -> int:  # noqa: ARG001
-        return threading.get_ident()
-
-    thread_ids = parallel_execute(
-        devices=[torch.device("cpu"), torch.device("meta")],
-        functions=[test_function, test_function],
-    )
-
-    current_thread_id = threading.get_ident()
-    for thread_id in thread_ids:
-        assert thread_id != current_thread_id
-
-
-def test__parallel_execute__multiple_devices__sets_is_parallel_to_True() -> None:
-    def test_function(device: torch.device, is_parallel: bool) -> bool:  # noqa: ARG001
-        return is_parallel
-
-    is_parallels = parallel_execute(
-        devices=[torch.device("cpu"), torch.device("meta")],
-        functions=[test_function, test_function],
-    )
-
-    assert list(is_parallels) == [True, True]
-
-
-def test__parallel_execute__multiple_devices__results_in_same_order_as_functions() -> (
-    None
-):
-    def a(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
-        return "a"
-
-    def b(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
-        return "b"
-
-    def c(device: torch.device, is_parallel: bool) -> str:  # noqa: ARG001
-        return "c"
-
-    results = parallel_execute(
-        devices=[torch.device("meta"), torch.device("meta")], functions=[a, b, c]
-    )
-
-    assert list(results) == ["a", "b", "c"]
diff --git a/tests/test_preprocessing.py b/tests/test_preprocessing.py
index 5cf38c5..a824373 100644
--- a/tests/test_preprocessing.py
+++ b/tests/test_preprocessing.py
@@ -1,30 +1,21 @@
 from __future__ import annotations
 
 import warnings
-from collections.abc import Callable
 from functools import partial
 
 import numpy as np
-import pandas as pd
 import pytest
 import torch
-from sklearn.compose import ColumnTransformer
-from sklearn.preprocessing import (
-    FunctionTransformer,
-    OneHotEncoder,
-    OrdinalEncoder,
-    PowerTransformer,
-)
+from sklearn.preprocessing import PowerTransformer
 
-from tabpfn import preprocessors
-from tabpfn.preprocessors import (
+from tabpfn.architectures.base import preprocessing
+from tabpfn.architectures.base.preprocessing import (
     AdaptiveQuantileTransformer,
     DifferentiableZNormStep,
     FeaturePreprocessingTransformerStep,
     ReshapeFeatureDistributionsStep,
     SafePowerTransformer,
 )
-from tabpfn.preprocessors.preprocessing_helpers import OrderPreservingColumnTransformer
 
 
 @pytest.fixture
@@ -47,7 +38,7 @@ def test_preprocessing_large_dataset():
         transform_name="quantile_norm",
         apply_to_categorical=False,
         append_to_original=False,
-        max_features_per_estimator=500,
+        subsample_features=-1,
         global_transformer_name=None,
         random_state=42,
     )
@@ -185,11 +176,9 @@ def test_diff_znorm_transform_with_zero_std(
         # Test 'auto' mode below the threshold: should append original features
         pytest.param("auto", 10, 20, id="auto_below_threshold_appends"),
         # Test 'auto' mode above the threshold: should NOT append original features
-        pytest.param("auto", 600, 500, id="auto_above_threshold_replaces"),
-        # If n features more than half of max_features_per_estimator we do not append
-        pytest.param("auto", 300, 300, id="auto_below_half_threshold_replaces"),
-        # True: always append after capping (600 ΓåÆ capped 500 ΓåÆ doubled)
-        pytest.param(True, 600, 1000, id="true_always_appends"),
+        pytest.param("auto", 600, 600, id="auto_above_threshold_replaces"),
+        # Test True: should always append, regardless of threshold
+        pytest.param(True, 600, 1200, id="true_always_appends"),
         # Test False: should never append
         pytest.param(False, 10, 10, id="false_never_appends"),
     ],
@@ -210,7 +199,6 @@ def test_reshape_step_append_original_logic(
         transform_name="quantile_norm",
         append_to_original=append_to_original_setting,
         random_state=42,
-        max_features_per_estimator=500,
     )
 
     # ACT: Run the preprocessing
@@ -221,12 +209,10 @@ def test_reshape_step_append_original_logic(
     assert Xt.shape[1] == expected_output_features
 
 
-def _get_preprocessing_steps() -> list[
-    Callable[..., FeaturePreprocessingTransformerStep],
-]:
-    defaults: list[Callable[..., FeaturePreprocessingTransformerStep]] = [
+def _get_preprocessing_steps():
+    defaults = [
         cls
-        for cls in preprocessors.__dict__.values()
+        for cls in preprocessing.__dict__.values()
         if (
             isinstance(cls, type)
             and issubclass(cls, FeaturePreprocessingTransformerStep)
@@ -234,7 +220,7 @@ def _get_preprocessing_steps() -> list[
             and cls is not DifferentiableZNormStep  # works on torch tensors
         )
     ]
-    extras: list[Callable[..., FeaturePreprocessingTransformerStep]] = [
+    extras = [
         partial(
             ReshapeFeatureDistributionsStep,
             transform_name="none",
@@ -246,9 +232,7 @@ def _get_preprocessing_steps() -> list[
     return defaults + extras
 
 
-def _get_random_data(
-    rng: np.random.Generator, n_samples: int, n_features: int, cat_inds: list[int]
-) -> np.ndarray:
+def _get_random_data(rng, n_samples, n_features, cat_inds):
     x = rng.random((n_samples, n_features))
     x[:, cat_inds] = rng.integers(0, 3, size=(n_samples, len(cat_inds))).astype(float)
     return x
@@ -295,16 +279,16 @@ def test__preprocessing_steps__transform__no_sample_interdependence():
         # Test 1: Shuffling samples should give correspondingly shuffled results
         result_normal = obj.transform(x2)
         result_reversed = obj.transform(x2[::-1])
-        assert np.allclose(result_reversed.X[::-1], result_normal.X), (
-            f"Transform depends on sample order for {cls}"
-        )
+        assert np.allclose(
+            result_reversed.X[::-1], result_normal.X
+        ), f"Transform depends on sample order for {cls}"
 
         # Test 2: Transforming a subset should match the subset of full transformation
         result_full = obj.transform(x2)
         result_subset = obj.transform(x2[:4])
-        assert np.allclose(result_full.X[:4], result_subset.X), (
-            f"Transform depends on other samples in batch for {cls}"
-        )
+        assert np.allclose(
+            result_full.X[:4], result_subset.X
+        ), f"Transform depends on other samples in batch for {cls}"
 
         # Test 3: Categorical features should remain the same
         assert result_full.categorical_features == result_subset.categorical_features
@@ -404,9 +388,9 @@ def test__safe_power_transformer__power_transformer_fails__no_error():
     )
 
     # check if result contains nan or inf
-    assert np.all(np.isfinite(safe_result)), (
-        "SafePowerTransformer produced non-finite values"
-    )
+    assert np.all(
+        np.isfinite(safe_result)
+    ), "SafePowerTransformer produced non-finite values"
 
 
 def test__safe_power_transformer__transform_then_inverse_transform__returns_original():
@@ -445,60 +429,3 @@ def test__safe_power_transformer__transform_then_inverse_transform__returns_orig
             atol=1e-7,
             err_msg=f"Inverse transform failed for test case {i}",
         )
-
-
-# This is a test for the OrderPreservingColumnTransformer, which is not used currently
-# But might be used in the future, therefore I'll leave it in.
-@pytest.mark.skip
-def test_order_preserving_column_transformer():
-    """Should raise AssertionError if column sets overlap."""
-    ordinal_enc1 = OrdinalEncoder()
-    ordinal_enc2 = OrdinalEncoder()
-    onehotencoder1 = OneHotEncoder()
-
-    # Test assertion raised due to too many transformers
-    multiple_transformers = [
-        ("ordinal_enc1", ordinal_enc1, ["a", "b"]),
-        ("ordinal_enc2", ordinal_enc2, ["c", "d"]),
-    ]
-
-    with pytest.raises(
-        AssertionError,
-        match="OrderPreservingColumnTransformer only supports up to one transformer",
-    ):
-        OrderPreservingColumnTransformer(transformers=multiple_transformers)
-
-    # Test assertion, due to unsupported encoder type (OneHotEncoder)
-    incompatible_transformer = [("onehot", onehotencoder1, ["a", "b"])]
-
-    with pytest.raises(AssertionError, match="are instances of OneToOneFeatureMixin"):
-        OrderPreservingColumnTransformer(transformers=incompatible_transformer)
-
-        # --- Mock dataset ---
-    mock_data_df = pd.DataFrame(
-        {
-            "a": [10, 20, 30, 40],
-            "b": ["x", "y", "x", "z"],
-        }
-    )
-
-    # Test if normal column transformer shuffles column order,
-    # while the OrderPreserving restores the original order
-    non_overlapping_ordinal_encoder = [("ordinal_enc1", ordinal_enc1, ["b"])]
-
-    vanilla_transformer = ColumnTransformer(
-        transformers=non_overlapping_ordinal_encoder, remainder=FunctionTransformer()
-    )
-
-    vanilla_output = vanilla_transformer.fit_transform(mock_data_df)
-
-    # Vanilla transformer shuffles column order
-    assert not np.array_equal(mock_data_df.iloc[:, 0].values, vanilla_output[:, 0])
-
-    preserving_transformer = OrderPreservingColumnTransformer(
-        transformers=non_overlapping_ordinal_encoder, remainder=FunctionTransformer()
-    )
-
-    # OrderPreserving transformer does not shuffle column order
-    preserved_output = preserving_transformer.fit_transform(mock_data_df)
-    np.testing.assert_equal(mock_data_df.iloc[:, 0].values, preserved_output[:, 0])
diff --git a/tests/test_regressor_interface.py b/tests/test_regressor_interface.py
index a10258f..ebaf56d 100644
--- a/tests/test_regressor_interface.py
+++ b/tests/test_regressor_interface.py
@@ -2,6 +2,7 @@ from __future__ import annotations
 
 import io
 import os
+import sys
 import typing
 from itertools import product
 from typing import Callable, Literal
@@ -20,20 +21,24 @@ from torch import nn
 
 from tabpfn import TabPFNRegressor
 from tabpfn.base import RegressorModelSpecs, initialize_tabpfn_model
-from tabpfn.constants import ModelVersion
-from tabpfn.model_loading import ModelSource
 from tabpfn.preprocessing import PreprocessorConfig
-from tabpfn.utils import infer_devices
+from tabpfn.utils import infer_device_and_type
 
-from .utils import check_cpu_float16_support, get_pytest_devices
+from .utils import check_cpu_float16_support
 
-devices = get_pytest_devices()
+exclude_devices = {
+    d.strip() for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",") if d.strip()
+}
+
+devices = ["cpu"]
+if torch.cuda.is_available() and "cuda" not in exclude_devices:
+    devices.append("cuda")
+if torch.backends.mps.is_available() and "mps" not in exclude_devices:
+    devices.append("mps")
 
 # --- Environment-Aware Check for CPU Float16 Support ---
 is_cpu_float16_supported = check_cpu_float16_support()
 
-# --- Define parameter combinations ---
-# These are the parameters we want to test in our grid search
 feature_shift_decoders = ["shuffle", "rotate"]
 fit_modes = [
     "low_memory",
@@ -44,44 +49,24 @@ inference_precision_methods = ["auto", "autocast", torch.float64, torch.float16]
 remove_outliers_stds = [None, 12]
 estimators = [1, 2]
 
-model_paths = ModelSource.get_regressor_v2().filenames
-primary_model = ModelSource.get_regressor_v2().default_filename
-other_models = [model_path for model_path in model_paths if model_path != primary_model]
-
-# --- Build parameter combinations ---
-# Full grid for the first (primary) model path
-_full_grid = product(
-    estimators,
-    devices,  # device
-    feature_shift_decoders,
-    fit_modes,
-    inference_precision_methods,
-    remove_outliers_stds,
-    [primary_model],  # only the first entry
-)
-
-# Minimal "smoke" grid for all remaining model paths (one combo per path)
-_smoke_grid = product(
-    [1],  # n_estimators
-    ["cpu"],  # device (fast & universally available)
-    ["shuffle"],  # feature_shift_decoder
-    ["fit_preprocessors"],  # fit_mode
-    ["auto"],  # inference_precision
-    [remove_outliers_stds[0]],  # remove_outliers_std
-    # every non-first model path and multiple models test
-    [*other_models, [primary_model, other_models[0]]],
+all_combinations = list(
+    product(
+        estimators,
+        devices,
+        feature_shift_decoders,
+        fit_modes,
+        inference_precision_methods,
+        remove_outliers_stds,
+    ),
 )
 
-all_combinations = list(_full_grid) + list(_smoke_grid)
-
 
 # Wrap in fixture so it's only loaded in if a test using it is run
 @pytest.fixture(scope="module")
 def X_y() -> tuple[np.ndarray, np.ndarray]:
-    X, y, _ = sklearn.datasets.make_regression(
-        n_samples=30, n_features=4, random_state=0, coef=True
-    )
-    return X, y
+    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
+    X, y = X[:40], y[:40]
+    return X, y  # type: ignore
 
 
 @pytest.mark.parametrize(
@@ -92,7 +77,6 @@ def X_y() -> tuple[np.ndarray, np.ndarray]:
         "fit_mode",
         "inference_precision",
         "remove_outliers_std",
-        "model_path",
     ),
     all_combinations,
 )
@@ -103,27 +87,22 @@ def test_regressor(
     fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
     inference_precision: torch.types._dtype | Literal["autocast", "auto"],
     remove_outliers_std: int | None,
-    model_path: str,
     X_y: tuple[np.ndarray, np.ndarray],
 ) -> None:
-    if inference_precision == "autocast":
-        if torch.device(device).type == "cpu":
-            pytest.skip("CPU device does not support 'autocast' inference.")
-        if torch.device(device).type == "mps" and torch.__version__ < "2.5":
-            pytest.skip("MPS does not support mixed precision before PyTorch 2.5")
+    if device == "cpu" and inference_precision == "autocast":
+        pytest.skip("Only GPU supports inference_precision")
 
     # Use the environment-aware check to skip only if necessary
     if (
-        torch.device(device).type == "cpu"
+        device == "cpu"
         and inference_precision == torch.float16
         and not is_cpu_float16_supported
     ):
         pytest.skip("CPU float16 matmul not supported in this PyTorch version.")
-    if torch.device(device).type == "mps" and inference_precision == torch.float64:
+    if device == "mps" and inference_precision == torch.float64:
         pytest.skip("MPS does not support float64, which is required for this check.")
 
     model = TabPFNRegressor(
-        model_path=model_path,
         n_estimators=n_estimators,
         device=device,
         fit_mode=fit_mode,
@@ -155,11 +134,9 @@ def test_regressor(
     assert quantiles[0].shape == (X.shape[0],), "Predictions shape is incorrect"
 
 
-# The different fitting modes manage the random state differently.
-@pytest.mark.skip(
-    reason="The prediction is actually different depending on the fitting mode."
-)
-def test_fit_modes_all_return_equal_results(X_y: tuple[np.ndarray, np.ndarray]) -> None:
+def test_fit_modes_all_return_equal_results(
+    X_y: tuple[np.ndarray, np.ndarray],
+) -> None:
     kwargs = {
         "n_estimators": 10,
         "device": "cpu",
@@ -176,42 +153,13 @@ def test_fit_modes_all_return_equal_results(X_y: tuple[np.ndarray, np.ndarray])
     torch.random.manual_seed(0)
     tabpfn = TabPFNRegressor(fit_mode="fit_with_cache", **kwargs)
     tabpfn.fit(X, y)
-    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X))
+    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X), decimal=5)
 
     torch.random.manual_seed(0)
     tabpfn = TabPFNRegressor(fit_mode="low_memory", **kwargs)
     tabpfn.fit(X, y)
-    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X))
-
-
-def test_multiple_models_predict_different_results(
-    X_y: tuple[np.ndarray, np.ndarray],
-):
-    """Tests the predict_raw_logits method."""
-    X, y = X_y
-
-    single_model = primary_model
-    two_identical_models = [primary_model, primary_model]
-    two_different_models = [primary_model, other_models[0]]
-
-    def get_prediction(model_paths: list[str]) -> np.ndarray:
-        regressor = TabPFNRegressor(
-            n_estimators=2,
-            random_state=42,
-            model_path=model_paths,
-        )
-        regressor.fit(X, y)
-        return regressor.predict(X)
-
-    single_model_pred = get_prediction(model_paths=[single_model])
-    two_identical_models_pred = get_prediction(model_paths=two_identical_models)
-    two_different_models_pred = get_prediction(model_paths=two_different_models)
-
-    assert not np.all(single_model_pred == single_model_pred[0:1]), (
-        "Logits are identical across classes for all samples, indicating trivial output"
-    )
-    assert np.all(single_model_pred == two_identical_models_pred)
-    assert not np.all(single_model_pred == two_different_models_pred)
+    # TODO: It's only equal to one decimal place. Verify if actually broken.
+    np.testing.assert_array_almost_equal(preds, tabpfn.predict(X), decimal=1)
 
 
 # TODO: Should probably run a larger suite with different configurations
@@ -220,8 +168,8 @@ def test_sklearn_compatible_estimator(
     estimator: TabPFNRegressor,
     check: Callable[[TabPFNRegressor], None],
 ) -> None:
-    _auto_devices = infer_devices(devices="auto")
-    if any(device.type == "mps" for device in _auto_devices):
+    _auto_device = infer_device_and_type(device="auto")
+    if _auto_device.type == "mps":
         pytest.skip("MPS does not support float64, which is required for this check.")
 
     if check.func.__name__ in (  # type: ignore
@@ -259,21 +207,21 @@ def test_regressor_in_pipeline(X_y: tuple[np.ndarray, np.ndarray]) -> None:
 
     # Test different prediction modes through the pipeline
     predictions_median = pipeline.predict(X, output_type="median")
-    assert predictions_median.shape == (X.shape[0],), (
-        "Median predictions shape is incorrect"
-    )
+    assert predictions_median.shape == (
+        X.shape[0],
+    ), "Median predictions shape is incorrect"
 
     predictions_mode = pipeline.predict(X, output_type="mode")
-    assert predictions_mode.shape == (X.shape[0],), (
-        "Mode predictions shape is incorrect"
-    )
+    assert predictions_mode.shape == (
+        X.shape[0],
+    ), "Mode predictions shape is incorrect"
 
     quantiles = pipeline.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
     assert isinstance(quantiles, list)
     assert len(quantiles) == 2
-    assert quantiles[0].shape == (X.shape[0],), (
-        "Quantile predictions shape is incorrect"
-    )
+    assert quantiles[0].shape == (
+        X.shape[0],
+    ), "Quantile predictions shape is incorrect"
 
 
 def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray]) -> None:
@@ -286,7 +234,7 @@ def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray])
         "append_original": False,  # changed from default
         "categorical_name": "ordinal_very_common_categories_shuffled",
         "global_transformer_name": "svd",
-        "max_features_per_estimator": 500,
+        "subsample_features": -1,
     }
 
     object_config = PreprocessorConfig(
@@ -294,7 +242,7 @@ def test_dict_vs_object_preprocessor_config(X_y: tuple[np.ndarray, np.ndarray])
         append_original=False,  # changed from default
         categorical_name="ordinal_very_common_categories_shuffled",
         global_transformer_name="svd",
-        max_features_per_estimator=500,
+        subsample_features=-1,
     )
 
     # Create two models with same random state
@@ -367,12 +315,12 @@ class ModelWrapper(nn.Module):
 def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     if os.name == "nt":
         pytest.skip("onnx export is not tested on windows")
+    if sys.version_info >= (3, 13):
+        pytest.xfail("onnx is not yet supported on Python 3.13")
     X, y = X_y
     with torch.no_grad():
-        regressor = TabPFNRegressor(
-            n_estimators=1, device="cpu", random_state=43, memory_saving_mode=True
-        )
-        # load the model so we can access it via classifier.models_
+        regressor = TabPFNRegressor(n_estimators=1, device="cpu", random_state=43)
+        # load the model so we can access it via classifier.model_
         regressor.fit(X, y)
         # this is necessary if cuda is available
         regressor.predict(X)
@@ -388,13 +336,8 @@ def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
             "X": {0: "num_datapoints", 1: "batch_size", 2: "num_features"},
             "y": {0: "num_labels"},
         }
-
-        # From 2.9 PyTorch changed the default export mode from TorchScript to
-        # Dynamo. We don't support Dynamo, so disable it. The `dynamo` flag is only
-        # available in newer PyTorch versions, hence we don't always include it.
-        export_kwargs = {"dynamo": False} if torch.__version__ >= "2.9" else {}
         torch.onnx.export(
-            ModelWrapper(regressor.models_[0]).eval(),
+            ModelWrapper(regressor.model_).eval(),
             (X, y, True, [[]]),
             io.BytesIO(),
             input_names=[
@@ -406,7 +349,6 @@ def test_onnx_exportable_cpu(X_y: tuple[np.ndarray, np.ndarray]) -> None:
             output_names=["output"],
             opset_version=17,  # using 17 since we use torch>=2.1
             dynamic_axes=dynamic_axes,
-            **export_kwargs,
         )
 
 
@@ -424,35 +366,31 @@ def test_get_embeddings(X_y: tuple[np.ndarray, np.ndarray], data_source: str) ->
     embeddings = model.get_embeddings(X, valid_data_source)
 
     # Need to access the model through the executor
-    model_instances = typing.cast(typing.Any, model.executor_).models
-    hidden_size = model_instances[0].ninp
+    model_instance = typing.cast(typing.Any, model.executor_).model
+    encoder_shape = next(
+        m.out_features
+        for m in model_instance.encoder.modules()
+        if isinstance(m, nn.Linear)
+    )
 
     assert isinstance(embeddings, np.ndarray)
     assert embeddings.shape[0] == n_estimators
     assert embeddings.shape[1] == X.shape[0]
-    assert embeddings.shape[2] == hidden_size
-
+    assert embeddings.shape[2] == encoder_shape
 
-def test_overflow_bug_does_not_occur():
-    """Test that an overflow does not occur in the preprocessing.
 
-    This can occur if scipy<1.11.0, see
-    https://github.com/PriorLabs/TabPFN/issues/175 .
-
-    It no longer appears to happen with the current preprocessing configuration, but
-    test just in case.
-    """
-    rng = np.random.default_rng(seed=0)
-    # This is a specially crafted dataset with nearly constant features that has been
-    # found to trigger the bug. The California housing dataset will also trigger it.
-    n = 20
-    X = 100.0 + rng.normal(loc=0.0, scale=0.0001, size=(n, 9))
-    y = rng.normal(loc=0.0, scale=1.0, size=(n,))
+def test_overflow():
+    """Test which fails for scipy<1.11.0."""
+    # Fetch a small sample of the California housing dataset
+    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
+    X, y = X[:20], y[:20]
 
+    # Create and fit the regressor
     regressor = TabPFNRegressor(n_estimators=1, device="cpu", random_state=42)
+
     regressor.fit(X, y)
-    predictions = regressor.predict(X)
 
+    predictions = regressor.predict(X)
     assert predictions.shape == (X.shape[0],), "Predictions shape is incorrect"
 
 
@@ -580,187 +518,94 @@ def test_constant_feature_handling(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     )
 
 
-@pytest.mark.parametrize("constant_value", [0.0, 1.0, -1.0, 1e-5, -1e-5, 1e5, -1e5])
-def test_constant_target(
-    X_y: tuple[np.ndarray, np.ndarray], constant_value: float
-) -> None:
+def test_constant_target(X_y: tuple[np.ndarray, np.ndarray]) -> None:
     """Test that TabPFNRegressor predicts a constant
-    value when the target y is constant, for both small and large values.
+    value when the target y is constant.
     """
     X, _ = X_y
-
-    y_constant = np.full(X.shape[0], constant_value)
+    y_constant = np.full(X.shape[0], 5.0)  # Create a constant target array
 
     model = TabPFNRegressor(n_estimators=2, random_state=42)
     model.fit(X, y_constant)
 
     predictions = model.predict(X)
-    assert np.all(predictions == constant_value), (
-        f"Predictions are not constant as expected for value {constant_value}"
-    )
+    assert np.all(predictions == 5.0), "Predictions are not constant as expected"
 
     # Test different output types
     predictions_median = model.predict(X, output_type="median")
-    assert np.all(predictions_median == constant_value), (
-        f"Median predictions are not constant as expected for value {constant_value}"
-    )
+    assert np.all(
+        predictions_median == 5.0
+    ), "Median predictions are not constant as expected"
 
     predictions_mode = model.predict(X, output_type="mode")
-    assert np.all(predictions_mode == constant_value), (
-        f"Mode predictions are not constant as expected for value {constant_value}"
-    )
+    assert np.all(
+        predictions_mode == 5.0
+    ), "Mode predictions are not constant as expected"
 
     quantiles = model.predict(X, output_type="quantiles", quantiles=[0.1, 0.9])
     for quantile_prediction in quantiles:
-        assert np.all(quantile_prediction == constant_value), (
-            f"Quantile predictions are not constant as expected for"
-            f" value {constant_value}"
-        )
+        assert np.all(
+            quantile_prediction == 5.0
+        ), "Quantile predictions are not constant as expected"
 
     full_output = model.predict(X, output_type="full")
-    assert np.all(full_output["mean"] == constant_value), (
-        f"Mean predictions are not constant as expected for full output for"
-        f" value {constant_value}"
-    )
-    assert np.all(full_output["median"] == constant_value), (
-        f"Median predictions are not constant as expected for full output for"
-        f" value {constant_value}"
-    )
-    assert np.all(full_output["mode"] == constant_value), (
-        f"Mode predictions are not constant as expected for full output for"
-        f" value {constant_value}"
-    )
+    assert np.all(
+        full_output["mean"] == 5.0
+    ), "Mean predictions are not constant as expected for full output"
+    assert np.all(
+        full_output["median"] == 5.0
+    ), "Median predictions are not constant as expected for full output"
+    assert np.all(
+        full_output["mode"] == 5.0
+    ), "Mode predictions are not constant as expected for full output"
     for quantile_prediction in full_output["quantiles"]:
-        assert np.all(quantile_prediction == constant_value), (
-            f"Quantile predictions are not constant as expected for full output for"
-            f" value {constant_value}"
-        )
+        assert np.all(
+            quantile_prediction == 5.0
+        ), "Quantile predictions are not constant as expected for full output"
 
 
 def test_initialize_model_variables_regressor_sets_required_attributes() -> None:
     # 1) Standalone initializer
-    model, architecture_configs, norm_criterion, inference_config = (
-        initialize_tabpfn_model(
-            model_path="auto",
-            which="regressor",
-            fit_mode="low_memory",
-        )
+    model, config, norm_criterion = initialize_tabpfn_model(
+        model_path="auto",
+        which="regressor",
+        fit_mode="low_memory",
     )
     assert model is not None, "model should be initialized for regressor"
-    assert architecture_configs is not None, (
-        "config should be initialized for regressor"
-    )
-    assert norm_criterion is not None, (
-        "norm_criterion should be initialized for regressor"
-    )
-    assert inference_config is not None
+    assert config is not None, "config should be initialized for regressor"
+    assert (
+        norm_criterion is not None
+    ), "norm_criterion should be initialized for regressor"
 
     # 2) Test the sklearn-style wrapper on TabPFNRegressor
-    regressor = TabPFNRegressor(device="cpu", random_state=42)
+    regressor = TabPFNRegressor(model_path="auto", device="cpu", random_state=42)
     regressor._initialize_model_variables()
 
-    assert hasattr(regressor, "models_")
-    assert regressor.models_ is not None
+    assert hasattr(regressor, "model_"), "regressor should have model_ attribute"
+    assert regressor.model_ is not None, "model_ should be initialized for regressor"
 
-    assert hasattr(regressor, "configs_")
-    assert regressor.configs_ is not None
+    assert hasattr(regressor, "config_"), "regressor should have config_ attribute"
+    assert regressor.config_ is not None, "config_ should be initialized for regressor"
 
-    assert hasattr(regressor, "znorm_space_bardist_")
-    assert regressor.znorm_space_bardist_ is not None
+    assert hasattr(regressor, "bardist_"), "regressor should have bardist_ attribute"
+    assert (
+        regressor.bardist_ is not None
+    ), "bardist_ should be initialized for regressor"
 
     # 3) Reuse via RegressorModelSpecs
     spec = RegressorModelSpecs(
-        model=regressor.models_[0],
-        architecture_config=regressor.configs_[0],
-        norm_criterion=regressor.znorm_space_bardist_,
-        inference_config=regressor.inference_config_,
+        model=regressor.model_,
+        config=regressor.config_,
+        norm_criterion=regressor.bardist_,
     )
     reg2 = TabPFNRegressor(model_path=spec)
     reg2._initialize_model_variables()
 
-    assert hasattr(reg2, "models_")
-    assert reg2.models_ is not None
-
-    assert hasattr(reg2, "configs_")
-    assert reg2.configs_ is not None
-
-    assert hasattr(reg2, "znorm_space_bardist_")
-    assert reg2.znorm_space_bardist_ is not None
-
-
-@pytest.mark.parametrize("n_features", [1, 2])
-def test__TabPFNRegressor__few_features__works(n_features: int) -> None:
-    """Test that TabPFNRegressor works correctly with 1 or 2 features."""
-    n_samples = 50
-
-    X, y, _ = sklearn.datasets.make_regression(
-        n_samples=n_samples,
-        n_features=n_features,
-        random_state=42,
-        coef=True,
-    )
-
-    model = TabPFNRegressor(
-        n_estimators=2,
-        random_state=42,
-    )
-
-    returned_model = model.fit(X, y)
-    assert returned_model is model, "Returned model is not the same as the model"
-    check_is_fitted(returned_model)
-
-    predictions = model.predict(X)
-    assert predictions.shape == (X.shape[0],), (
-        f"Predictions shape is incorrect for {n_features} features"
-    )
-    assert not np.isnan(predictions).any(), "Predictions contain NaN values"
-    assert not np.isinf(predictions).any(), "Predictions contain infinite values"
-
-    predictions_median = model.predict(X, output_type="median")
-    assert predictions_median.shape == (X.shape[0],), (
-        f"Median predictions shape is incorrect for {n_features} features"
-    )
-
-    predictions_mode = model.predict(X, output_type="mode")
-    assert predictions_mode.shape == (X.shape[0],), (
-        f"Mode predictions shape is incorrect for {n_features} features"
-    )
-
-    quantiles = model.predict(X, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
-    assert isinstance(quantiles, list), "Quantiles should be returned as a list"
-    assert len(quantiles) == 3, "Should return 3 quantiles"
-    for i, q in enumerate(quantiles):
-        assert q.shape == (X.shape[0],), (
-            f"Quantile {i} shape is incorrect for {n_features} features"
-        )
-
-
-def test__create_default_for_version__v2__uses_correct_defaults() -> None:
-    estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
-
-    assert isinstance(estimator, TabPFNRegressor)
-    assert estimator.n_estimators == 8
-    assert estimator.softmax_temperature == 0.9
-    assert isinstance(estimator.model_path, str)
-    assert "regressor" in estimator.model_path
-    assert "-v2-" in estimator.model_path
+    assert hasattr(reg2, "model_"), "regressor2 should have model_ attribute"
+    assert reg2.model_ is not None, "model_ should be initialized for regressor2"
 
+    assert hasattr(reg2, "config_"), "regressor2 should have config_ attribute"
+    assert reg2.config_ is not None, "config_ should be initialized for regressor2"
 
-def test__create_default_for_version__v2_5__uses_correct_defaults() -> None:
-    estimator = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
-
-    assert isinstance(estimator, TabPFNRegressor)
-    assert estimator.n_estimators == 8
-    assert estimator.softmax_temperature == 0.9
-    assert isinstance(estimator.model_path, str)
-    assert "regressor" in estimator.model_path
-    assert "-v2.5-" in estimator.model_path
-
-
-def test__create_default_for_version__passes_through_overrides() -> None:
-    estimator = TabPFNRegressor.create_default_for_version(
-        ModelVersion.V2_5, n_estimators=16
-    )
-
-    assert estimator.n_estimators == 16
-    assert estimator.softmax_temperature == 0.9
+    assert hasattr(reg2, "bardist_"), "regressor2 should have bardist_ attribute"
+    assert reg2.bardist_ is not None, "bardist_ should be initialized for regressor2"
diff --git a/tests/test_save_load_fitted_model.py b/tests/test_save_load_fitted_model.py
index ba6bf67..088b079 100644
--- a/tests/test_save_load_fitted_model.py
+++ b/tests/test_save_load_fitted_model.py
@@ -1,37 +1,11 @@
 # tests/test_save_load_fitted_model.py
 from __future__ import annotations
 
-from copy import deepcopy
-from itertools import product
-from pathlib import Path
-
 import numpy as np
 import pytest
-import torch
 from sklearn.datasets import make_classification, make_regression
 
 from tabpfn import TabPFNClassifier, TabPFNRegressor
-from tabpfn.architectures.interface import ArchitectureConfig
-from tabpfn.base import RegressorModelSpecs, initialize_tabpfn_model
-from tabpfn.inference_tuning import ClassifierEvalMetrics
-from tabpfn.model_loading import save_tabpfn_model
-
-from .utils import get_pytest_devices
-
-# filter out combinations when "mps" is exatly one device type!
-# -> yields different predictions, as dtypes are partly unsupported
-device_bicombination = [
-    comb for comb in product(get_pytest_devices(), repeat=2) if comb.count("mps") != 1
-]
-
-
-# --- Fixtures for cuda availability ---
-@pytest.fixture
-def disable_cuda_temporarily():
-    """Temporarily disable CUDA for a test."""
-    original_is_available = torch.cuda.is_available  # Cache original
-    yield
-    torch.cuda.is_available = original_is_available  # Restore after test
 
 
 # --- Fixtures for data ---
@@ -54,100 +28,41 @@ def classification_data_with_categoricals():
 
 # --- Main Test using Parametrization ---
 @pytest.mark.parametrize(
-    ("estimator_class", "data_fixture", "saving_device", "loading_device"),
+    ("estimator_class", "data_fixture"),
     [
-        (pred, fixture, *devs)
-        for (pred, fixture), devs in product(
-            [
-                (TabPFNRegressor, "regression_data"),
-                (TabPFNClassifier, "classification_data_with_categoricals"),
-            ],
-            device_bicombination,
-        )
+        (TabPFNRegressor, "regression_data"),
+        (TabPFNClassifier, "classification_data_with_categoricals"),
     ],
 )
-def test_save_load_happy_path(
-    estimator_class: type[TabPFNRegressor] | type[TabPFNClassifier],
-    data_fixture: str,
-    saving_device: str,
-    loading_device: str,
-    request: pytest.FixtureRequest,
-    tmp_path: Path,
-    monkeypatch: pytest.MonkeyPatch,
-) -> None:
+def test_save_load_happy_path(estimator_class, data_fixture, request, tmp_path):
+    """Tests the standard save/load workflow, including categorical data."""
     X, y = request.getfixturevalue(data_fixture)
-
-    # Simulate saving device
-    if "cuda" in saving_device:
-        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
-        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
-    elif "mps" in saving_device:
-        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
-        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
-    elif "cpu" in saving_device:
-        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
-        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
-    else:
-        raise NotImplementedError(f"saving device: {saving_device} not found")
-
-    model = estimator_class(device=saving_device, n_estimators=4)
+    model = estimator_class(device="cpu", n_estimators=4)
     model.fit(X, y)
     path = tmp_path / "model.tabpfn_fit"
 
     # Save and then load the model using its class method
     model.save_fit_state(path)
+    loaded_model = estimator_class.load_from_fit_state(path, device="cpu")
 
-    # Simulate saving device
-    if "cuda" in loading_device:
-        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
-        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
-    elif "mps" in loading_device:
-        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
-        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
-    elif "cpu" in loading_device:
-        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
-        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
-    else:
-        raise NotImplementedError(f"saving device: {loading_device} not found")
-
-    loaded_model = estimator_class.load_from_fit_state(path, device=loading_device)
-
-    if loading_device == saving_device:
-        # In transformer.py::add_embeddings we generate() random tensors inside a
-        # fixed-seed RNG context.
-        # Note: PyTorch uses different random number generators on CPU and CUDA.
-        # Even with the same seed, CPU and CUDA will produce different random values.
-        # This means the feature embeddings differ slightly depending on the device,
-        # which in turn leads to small prediction differences between CPU and CUDA
-        # models.
-        # This behavior is expected and comes from the transformer architecture,
-        # not a bug.
-
-        # We cannot align the two RNG streams, so the only options are either to skip
-        # the tests that compare predictions of different saving & loading devices.
+    # 1. Check that predictions are identical
+    np.testing.assert_array_almost_equal(model.predict(X), loaded_model.predict(X))
 
-        # (or use a large tolerance, which is reasonable for different random embeddings
-        # but as the regressor has a difference of +-1 unit, setting such a large
-        # tolerance is meaningless)
-
-        # 1. Check that predictions are identical
-        np.testing.assert_array_almost_equal(model.predict(X), loaded_model.predict(X))
-
-        # 2. For classifiers, also check probabilities and restored classes
-        if hasattr(model, "predict_proba"):
-            np.testing.assert_array_almost_equal(
-                model.predict_proba(X),
-                loaded_model.predict_proba(X),
-            )
-            np.testing.assert_array_equal(model.classes_, loaded_model.classes_)
+    # 2. For classifiers, also check probabilities and restored classes
+    if hasattr(model, "predict_proba"):
+        np.testing.assert_array_almost_equal(
+            model.predict_proba(X), loaded_model.predict_proba(X)
+        )
+        np.testing.assert_array_equal(model.classes_, loaded_model.classes_)
 
     # 3. Check that the loaded object is of the correct type
     assert isinstance(loaded_model, estimator_class)
 
 
 # --- Error Handling Tests ---
-def test_saving_unfitted_model_raises_error(tmp_path: Path) -> None:
+def test_saving_unfitted_model_raises_error(regression_data, tmp_path):
     """Tests that saving an unfitted model raises a RuntimeError."""
+    X, y = regression_data
     model = TabPFNRegressor()
     with pytest.raises(RuntimeError, match="Estimator must be fitted before saving"):
         model.save_fit_state(tmp_path / "model.tabpfn_fit")
@@ -165,132 +80,3 @@ def test_loading_mismatched_types_raises_error(regression_data, tmp_path):
         TypeError, match="Attempting to load a 'TabPFNRegressor' as 'TabPFNClassifier'"
     ):
         TabPFNClassifier.load_from_fit_state(path)
-
-
-def _init_and_save_unique_checkpoint(
-    model: TabPFNRegressor | TabPFNClassifier,
-    save_path: Path,
-) -> tuple[torch.Tensor, ArchitectureConfig]:
-    model._initialize_model_variables()
-    first_param = next(model.models_[0].parameters())
-    with torch.no_grad():
-        first_param.copy_(torch.randn_like(first_param))
-    first_model_parameter = first_param.clone()
-    config_before_saving = deepcopy(model.configs_[0])
-    save_tabpfn_model(model, save_path)
-
-    return first_model_parameter, config_before_saving
-
-
-def test_saving_and_loading_model_with_weights(tmp_path: Path) -> None:
-    """Tests that the saving format of the `save_tabpfn_model` method is compatible with
-    the loading interface of `initialize_tabpfn_model`.
-    """
-    # initialize a TabPFNRegressor
-    regressor = TabPFNRegressor(model_path="auto", device="cpu", random_state=42)
-    save_path = tmp_path / "model.ckpt"
-    first_model_parameter, config_before_saving = _init_and_save_unique_checkpoint(
-        model=regressor,
-        save_path=save_path,
-    )
-
-    # Load the model state
-    models, architecture_configs, criterion, inference_config = initialize_tabpfn_model(
-        save_path, "regressor", fit_mode="low_memory"
-    )
-    loaded_regressor = TabPFNRegressor(
-        model_path=RegressorModelSpecs(
-            model=models[0],
-            architecture_config=architecture_configs[0],
-            norm_criterion=criterion,
-            inference_config=inference_config,
-        ),
-        device="cpu",
-    )
-
-    # then check the model is loaded correctly
-    loaded_regressor._initialize_model_variables()
-    torch.testing.assert_close(
-        next(loaded_regressor.models_[0].parameters()),
-        first_model_parameter,
-    )
-    assert loaded_regressor.configs_[0] == config_before_saving
-
-
-@pytest.mark.parametrize(
-    ("estimator_class"),
-    [TabPFNRegressor, TabPFNClassifier],
-)
-def test_saving_and_loading_multiple_models_with_weights(
-    estimator_class: type[TabPFNRegressor] | type[TabPFNClassifier],
-    tmp_path: Path,
-) -> None:
-    """Test that saving and loading multiple models works."""
-    estimator = estimator_class(model_path="auto", device="cpu", random_state=42)
-    save_path_0 = tmp_path / "model_0.ckpt"
-    first_model_parameter_0, config_before_saving_0 = _init_and_save_unique_checkpoint(
-        model=estimator,
-        save_path=save_path_0,
-    )
-    estimator = estimator_class(model_path="auto", device="cpu", random_state=42)
-    save_path_1 = tmp_path / "model_1.ckpt"
-    first_model_parameter_1, config_before_saving_1 = _init_and_save_unique_checkpoint(
-        model=estimator,
-        save_path=save_path_1,
-    )
-
-    loaded_estimator = estimator_class(
-        model_path=[save_path_0, save_path_1],
-        device="cpu",
-        random_state=42,
-    )
-    loaded_estimator._initialize_model_variables()
-
-    torch.testing.assert_close(
-        next(loaded_estimator.models_[0].parameters()),
-        first_model_parameter_0,
-    )
-    torch.testing.assert_close(
-        next(loaded_estimator.models_[1].parameters()),
-        first_model_parameter_1,
-    )
-    assert loaded_estimator.configs_[0] == config_before_saving_0
-    assert loaded_estimator.configs_[1] == config_before_saving_1
-
-    with pytest.raises(ValueError, match="Your TabPFN estimator has multiple"):
-        save_tabpfn_model(loaded_estimator, Path(tmp_path) / "DOES_NOT_SAVE.ckpt")
-
-    save_tabpfn_model(
-        loaded_estimator,
-        [Path(tmp_path) / "0.ckpt", Path(tmp_path) / "1.ckpt"],
-    )
-    assert (tmp_path / "0.ckpt").exists()
-    assert (tmp_path / "1.ckpt").exists()
-
-
-def test_saving_and_loading_with_tuning_config(
-    tmp_path: Path,
-) -> None:
-    """Test that saving and loading a model with a tuning config works."""
-    estimator = TabPFNClassifier(
-        device="cpu",
-        random_state=42,
-        eval_metric="f1",
-        # TODO: test the case when dataclass is used
-        tuning_config={
-            "tune_decision_thresholds": True,
-            "calibrate_temperature": True,
-            "tuning_holdout_frac": 0.1,
-            "tuning_n_folds": 1,
-        },
-    )
-    X, y = make_classification(
-        n_samples=50, n_features=5, n_classes=3, n_informative=3, random_state=42
-    )
-    path = tmp_path / "model.tabpfn_fit"
-    estimator.fit(X, y)
-    estimator.save_fit_state(path)
-    loaded_estimator = TabPFNClassifier.load_from_fit_state(path)
-    assert loaded_estimator.tuned_classification_thresholds_ is not None
-    assert loaded_estimator.softmax_temperature_ is not None
-    assert loaded_estimator.eval_metric_ is ClassifierEvalMetrics.F1
diff --git a/tests/test_scripts.py b/tests/test_scripts.py
new file mode 100644
index 0000000..548d126
--- /dev/null
+++ b/tests/test_scripts.py
@@ -0,0 +1,77 @@
+# tests/test_scripts.py
+
+from __future__ import annotations
+
+import sys
+from pathlib import Path
+
+import pytest
+
+# 1. Import the main function from the NEW, merged script
+from scripts.generate_dependencies import main as generate_deps_main
+
+# --- Shared Test Data (This part is unchanged) ---
+
+PYPROJECT_CONTENT = """
+[project]
+name = "my-package"
+dependencies = [
+  "torch>=1.0,<3",
+  "scipy<2.0",
+  "pandas>=1.4.0",
+  "einops",
+  "huggingface-hub>=0.0.1,<1",
+]
+"""
+EXPECTED_MAX_REQS = sorted(
+    ["torch<3", "scipy<2.0", "pandas", "einops", "huggingface-hub<1"]
+)
+EXPECTED_MIN_REQS = sorted(["torch==1.0", "pandas==1.4.0", "huggingface-hub==0.0.1"])
+
+
+# 2. Update the parametrize decorator to pass the 'mode' string instead of a function
+@pytest.mark.parametrize(
+    ("mode", "expected_requirements"),
+    [
+        ("maximum", EXPECTED_MAX_REQS),
+        ("minimum", EXPECTED_MIN_REQS),
+    ],
+)
+def test_dependency_script_generation(
+    mode: str,
+    expected_requirements: list[str],
+    tmp_path: Path,
+    monkeypatch: pytest.MonkeyPatch,
+) -> None:
+    """Tests the unified generate_dependencies.py script for both 'min' and 'max' modes.
+
+    This test simulates the command-line arguments required by the new script.
+    """
+    # ARRANGE
+    # This part is the same: create the pyproject.toml and change directory
+    p = tmp_path / "pyproject.toml"
+    p.write_text(PYPROJECT_CONTENT)
+    monkeypatch.chdir(tmp_path)
+
+    # 3. Simulate command-line arguments using monkeypatch.
+    #    We are setting sys.argv to what it would be if called from the terminal,
+    #    e.g., ['generate_dependencies.py', 'min']
+    monkeypatch.setattr(sys, "argv", ["generate_dependencies.py", mode])
+
+    # ACT
+    # Call the single main function. It will now parse the mocked sys.argv.
+    generate_deps_main()
+
+    # ASSERT
+    # The assertion logic remains exactly the same.
+    output_file = tmp_path / "requirements.txt"
+    assert (
+        output_file.is_file()
+    ), f"Script did not create requirements.txt in '{mode}' mode."
+
+    with output_file.open() as f:
+        actual_requirements = sorted(line.strip() for line in f if line.strip())
+
+    assert (
+        actual_requirements == expected_requirements
+    ), f"Output in '{mode}' mode did not match expectations."
diff --git a/tests/test_settings.py b/tests/test_settings.py
deleted file mode 100644
index dd1374e..0000000
--- a/tests/test_settings.py
+++ /dev/null
@@ -1,31 +0,0 @@
-from __future__ import annotations
-
-from pathlib import Path
-
-import pytest
-
-from tabpfn.settings import TabPFNSettings, TestingSettings
-
-
-def test__load_settings__env_file_contains_variables_for_other_apps__does_not_crash(
-    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
-) -> None:
-    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
-    env_file = tmp_path / ".env"
-    with env_file.open(mode="w") as f:
-        f.write("OTHER_APP_VAR=1\n")
-        f.write("TABPFN_MODEL_CACHE_DIR=test_cache_dir\n")
-    tabpfn_settings = TabPFNSettings(_env_file=env_file)
-    assert str(tabpfn_settings.model_cache_dir) == "test_cache_dir"
-
-
-def test__ci_env_non_boolean_sets_true(monkeypatch: pytest.MonkeyPatch) -> None:
-    monkeypatch.setenv("CI", "azure")
-    testing_settings = TestingSettings()
-    assert testing_settings.ci is True
-
-
-def test__ci_env_false_sets_false(monkeypatch: pytest.MonkeyPatch) -> None:
-    monkeypatch.setenv("CI", "false")
-    testing_settings = TestingSettings()
-    assert testing_settings.ci is False
diff --git a/tests/test_telemetry_disabled.py b/tests/test_telemetry_disabled.py
deleted file mode 100644
index 9f8f3a8..0000000
--- a/tests/test_telemetry_disabled.py
+++ /dev/null
@@ -1,13 +0,0 @@
-"""Test to verify that telemetry is disabled during test execution."""
-
-from __future__ import annotations
-
-import os
-
-
-def test_telemetry_disabled():
-    """Verify that TABPFN_DISABLE_TELEMETRY is set to 1 during tests."""
-    assert os.environ.get("TABPFN_DISABLE_TELEMETRY") == "1", (
-        "TABPFN_DISABLE_TELEMETRY should be set to '1' during test execution. "
-        "This ensures telemetry is disabled for all tests."
-    )
diff --git a/tests/test_thinking_tokens.py b/tests/test_thinking_tokens.py
deleted file mode 100644
index 16e7c09..0000000
--- a/tests/test_thinking_tokens.py
+++ /dev/null
@@ -1,79 +0,0 @@
-from __future__ import annotations
-
-import torch
-
-from tabpfn.architectures.base.thinking_tokens import AddThinkingTokens
-
-
-def test__forward__output_has_correct_shape() -> None:
-    emsize = 8
-    module = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)
-
-    batch_size = 2
-    rows = 10
-    features = 3
-    embedded_input = torch.randn(batch_size, rows, features, emsize)
-    single_eval_pos = 7
-
-    output, new_single_eval_pos = module(embedded_input, single_eval_pos)
-
-    assert output.shape == (
-        batch_size,
-        15,  # rows + num_thinking_rows
-        features,
-        emsize,
-    )
-    assert new_single_eval_pos == 12  # original + num_thinking_rows
-
-
-def test__forward__tokens_equal_for_each_feature() -> None:
-    emsize = 8
-    module = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)
-
-    batch_size = 2
-    n_rows = 10
-    n_features = 3
-    embedded_input = torch.randn(batch_size, n_rows, n_features, emsize)
-    single_eval_pos = 7
-
-    output, _ = module(embedded_input, single_eval_pos)
-
-    assert output[0, 0, 0, 0] == output[0, 0, 1, 0]
-    assert output[0, 0, 0, 0] == output[0, 0, 2, 0]
-    assert output[0, 1, 0, 0] == output[0, 1, 1, 0]
-    assert output[0, 1, 0, 0] == output[0, 1, 2, 0]
-
-
-def test__forward__tokens_different_for_each_row() -> None:
-    emsize = 8
-    module = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)
-
-    batch_size = 2
-    n_rows = 3
-    n_features = 3
-    embedded_input = torch.randn(batch_size, n_rows, n_features, emsize)
-    single_eval_pos = 7
-
-    output, _ = module(embedded_input, single_eval_pos)
-
-    assert not torch.allclose(output[0, 0, 0, 0], output[0, 1, 0, 0])
-    assert not torch.allclose(output[0, 0, 0, 0], output[0, 2, 0, 0])
-    assert not torch.allclose(output[0, 0, 0, 0], output[0, 1, 0, 1])
-    assert not torch.allclose(output[0, 0, 0, 0], output[0, 2, 0, 1])
-
-
-def test__save_and_load__output_has_same_value() -> None:
-    emsize = 16
-    embedded_input = torch.randn(2, 10, 3, emsize)
-    single_eval_pos = 7
-
-    module_1 = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)
-    module_2 = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)
-
-    output_1, new_pos_1 = module_1(embedded_input, single_eval_pos)
-    state = module_1.state_dict()
-    module_2.load_state_dict(state)
-    output_2, new_pos_2 = module_2(embedded_input, single_eval_pos)
-
-    assert new_pos_1 == new_pos_2
-    assert torch.allclose(output_1, output_2)
diff --git a/tests/test_transformer.py b/tests/test_transformer.py
deleted file mode 100644
index 7c8b4af..0000000
--- a/tests/test_transformer.py
+++ /dev/null
@@ -1,35 +0,0 @@
-from __future__ import annotations
-
-from dataclasses import replace
-
-import torch
-
-from tabpfn.architectures.base.config import ModelConfig
-from tabpfn.architectures.base.transformer import PerFeatureTransformer
-
-
-def test__forward__thinking_rows_enabled__output_has_correct_shape() -> None:
-    config = replace(_minimal_config(), num_thinking_rows=5)
-    model = PerFeatureTransformer(config=config)
-    batch = 2
-    train_rows = 10
-    eval_rows = 3
-    total_rows = train_rows + eval_rows
-    x_features = 2
-    y_features = 1
-    x = {"main": torch.randn(total_rows, batch, x_features)}
-    y = {"main": torch.randn(train_rows, batch, y_features)}
-    output = model(x, y)
-    assert output.shape == (eval_rows, batch, 1)
-
-
-def _minimal_config() -> ModelConfig:
-    return ModelConfig(
-        emsize=8,
-        features_per_group=1,
-        max_num_classes=10,
-        nhead=2,
-        nlayers=2,
-        remove_duplicate_features=True,
-        num_buckets=1000,
-    )
diff --git a/tests/test_utils.py b/tests/test_utils.py
index 8a62821..4b513f7 100644
--- a/tests/test_utils.py
+++ b/tests/test_utils.py
@@ -1,361 +1,45 @@
+# use get_total_memory and compare it against result from psutils
+# run it only if the it is windows os.name == "nt"
 from __future__ import annotations
 
-from unittest.mock import MagicMock
+import os
 
-import numpy as np
-import pandas as pd
 import pytest
-import torch
-from sklearn.preprocessing import LabelEncoder
 
-from tabpfn import TabPFNClassifier
-from tabpfn.constants import NA_PLACEHOLDER
-from tabpfn.inference_config import InferenceConfig
-from tabpfn.preprocessors.preprocessing_helpers import get_ordinal_encoder
-from tabpfn.utils import (
-    balance_probas_by_class_counts,
-    fix_dtypes,
-    infer_categorical_features,
-    infer_devices,
-    process_text_na_dataframe,
-    validate_Xy_fit,
-)
 
+def test_internal_windows_total_memory():
+    if os.name != "nt":
+        pytest.skip("Windows specific test")
+    import psutil
 
-def test_infer_categorical_with_str_and_nan_provided_included():
-    X = np.array([[np.nan, "NA"]], dtype=object).reshape(-1, 1)
-    out = infer_categorical_features(
-        X,
-        provided=[0],
-        min_samples_for_inference=0,
-        max_unique_for_category=2,
-        min_unique_for_numerical=5,
-    )
-    assert out == [0]
+    from tabpfn.utils import get_total_memory_windows
 
+    utils_result = get_total_memory_windows()
+    psutil_result = psutil.virtual_memory().total / 1e9
+    assert utils_result == psutil_result
 
-def test_infer_categorical_with_str_and_nan_multiple_rows_provided_included():
-    X = np.array([[np.nan], ["NA"], ["NA"]], dtype=object)
-    out = infer_categorical_features(
-        X,
-        provided=[0],
-        min_samples_for_inference=0,
-        max_unique_for_category=2,
-        min_unique_for_numerical=5,
-    )
-    assert out == [0]
 
+def test_internal_windows_total_memory_multithreaded():
+    # collect results from multiple threads
+    if os.name != "nt":
+        pytest.skip("Windows specific test")
+    import threading
 
-def test_infer_categorical_auto_inference_blocked_when_not_enough_samples():
-    X = np.array([[1.0], [1.0], [np.nan]])
-    out = infer_categorical_features(
-        X,
-        provided=None,
-        min_samples_for_inference=3,
-        max_unique_for_category=2,
-        min_unique_for_numerical=4,
-    )
-    assert out == []
+    import psutil
 
+    from tabpfn.utils import get_total_memory_windows
 
-def test_infer_categorical_auto_inference_enabled_with_enough_samples():
-    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0], [np.nan, 9.0]])
-    out = infer_categorical_features(
-        X,
-        provided=None,
-        min_samples_for_inference=3,
-        max_unique_for_category=3,
-        min_unique_for_numerical=4,
-    )
-    assert out == [0]
+    results = []
 
+    def get_memory():
+        results.append(get_total_memory_windows())
 
-def test_infer_categorical_provided_column_excluded_if_exceeds_max_unique():
-    X = np.array([[0], [1], [2], [3], [np.nan]], dtype=float)
-    out = infer_categorical_features(
-        X,
-        provided=[0],
-        min_samples_for_inference=0,
-        max_unique_for_category=3,
-        min_unique_for_numerical=2,
-    )
-    assert out == []
-
-
-def test_infer_categorical_with_dict_raises_error():
-    X = np.array([[{"a": 1}], [{"b": 2}]], dtype=object)
-    with pytest.raises(TypeError):
-        infer_categorical_features(
-            X,
-            provided=None,
-            min_samples_for_inference=0,
-            max_unique_for_category=2,
-            min_unique_for_numerical=2,
-        )
-
-
-def test__infer_devices__auto__cuda_and_mps_not_available__selects_cpu(
-    mocker: MagicMock,
-) -> None:
-    mocker.patch("torch.cuda").is_available.return_value = False
-    mocker.patch("torch.backends.mps").is_available.return_value = False
-    assert infer_devices(devices="auto") == (torch.device("cpu"),)
-
-
-def test__infer_devices__auto__single_cuda_gpu_available__selects_it(
-    mocker: MagicMock,
-) -> None:
-    mock_cuda = mocker.patch("torch.cuda")
-    mock_cuda.is_available.return_value = True
-    mock_cuda.device_count.return_value = 1
-    mocker.patch("torch.backends.mps").is_available.return_value = True
-    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)
-
-
-def test__infer_devices__auto__multiple_cuda_gpus_available__selects_first(
-    mocker: MagicMock,
-) -> None:
-    mock_cuda = mocker.patch("torch.cuda")
-    mock_cuda.is_available.return_value = True
-    mock_cuda.device_count.return_value = 3
-    mocker.patch("torch.backends.mps").is_available.return_value = True
-
-    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)
-
-
-def test__infer_devices__auto__cuda_and_mps_available_but_excluded__selects_cpu(
-    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
-) -> None:
-    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "mps,cuda")
-    mock_cuda = mocker.patch("torch.cuda")
-    mock_cuda.is_available.return_value = True
-    mock_cuda.device_count.return_value = 1
-    mocker.patch("torch.backends.mps").is_available.return_value = True
-    assert infer_devices(devices="auto") == (torch.device("cpu"),)
-
-
-def test__infer_devices__device_specified__selects_it(
-    mocker: MagicMock,
-) -> None:
-    mock_cuda = mocker.patch("torch.cuda")
-    mock_cuda.is_available.return_value = True
-    mock_cuda.device_count.return_value = 2
-    mocker.patch("torch.backends.mps").is_available.return_value = True
-
-    assert infer_devices(devices="cuda:0") == (torch.device("cuda:0"),)
-
-
-def test__infer_devices__multiple_devices_specified___selects_them(
-    mocker: MagicMock,
-) -> None:
-    mock_cuda = mocker.patch("torch.cuda")
-    mock_cuda.is_available.return_value = True
-    mock_cuda.device_count.return_value = 3
-    mocker.patch("torch.backends.mps").is_available.return_value = False
-
-    inferred = set(infer_devices(devices=["cuda:0", "cuda:1", "cuda:4"]))
-    expected = {torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:4")}
-    assert inferred == expected
-
-
-def test__infer_devices__device_selected_twice__raises() -> None:
-    with pytest.raises(
-        ValueError,
-        match="The list of devices for inference cannot contain the same device more ",
-    ):
-        infer_devices(devices=["cpu", "cpu"])
-
-
-# --- Test Data for the "test_process_text_na_dataframe" test ---
-test_cases = [
-    {
-        # Mixed dtypes & None / pd.Na
-        "df": pd.DataFrame(
-            {
-                "ratio": [0.4, 0.5, 0.6],
-                "risk": ["High", None, "Low"],
-                "height": ["Low", "Low", "Low"],
-                "amount": [10.2, 20.4, 20.5],
-                "type": ["guest", "member", pd.NA],
-            }
-        ),
-        "categorical_indices": [1, 2, 4],
-        "ground_truth": np.array(
-            [
-                [0.4, 0, 0, 10.2, 0],
-                [0.5, np.nan, 0, 20.4, 1],
-                [0.6, 1, 0, 20.5, np.nan],
-            ]
-        ),
-    },
-    {
-        # Mixed dtypes & no missing values
-        "df": pd.DataFrame(
-            {
-                "ratio": [0.4, 0.5, 0.6],
-                "risk": ["High", "Medium", "Low"],
-                "height": ["Low", "Low", "High"],
-                "amount": [10.2, 20.4, np.nan],
-                "type": ["guest", "member", "vip"],
-            }
-        ),
-        "categorical_indices": [1, 2, 4],
-        "ground_truth": np.array(
-            [
-                [0.4, 0, 1, 10.2, 0],
-                [0.5, 2, 1, 20.4, 1],
-                [0.6, 1, 0, np.nan, 2],
-            ]
-        ),
-    },
-    {
-        # All numerical no nan
-        "df": pd.DataFrame(
-            {
-                "ratio": [0.1, 0.2, 0.3],
-                "amount": [5.0, 15.5, 25.0],
-                "score": [1.0, 2.5, 3.5],
-            }
-        ),
-        "categorical_indices": [],
-        "ground_truth": np.array(
-            [
-                [0.1, 5.0, 1.0],
-                [0.2, 15.5, 2.5],
-                [0.3, 25.0, 3.5],
-            ]
-        ),
-    },
-    {
-        # all categorical no nan
-        "df": pd.DataFrame(
-            {
-                "risk": ["High", "High", "High"],
-                "height": ["Low", "Low", "Low"],
-                "type": ["guest", "guest", "guest"],
-            }
-        ),
-        "categorical_indices": [0, 1, 2],
-        "ground_truth": np.array(
-            [
-                [0, 0, 0],
-                [0, 0, 0],
-                [0, 0, 0],
-            ]
-        ),
-    },
-]
-
-
-# --- Fixture for the "test_process_text_na_dataframe" test ---
-# prepare the DataFrame
-@pytest.fixture(params=test_cases)
-def prepared_tabpfn_data(request):
-    temp_df = request.param["df"].copy()
-    categorical_idx = request.param["categorical_indices"]
-    # Dummy target, as tests do not need a target
-    y = np.array([0, 1, 0])
-
-    cls = TabPFNClassifier()
-
-    X, y, feature_names_in, n_features_in = validate_Xy_fit(
-        temp_df,
-        y,
-        estimator=cls,
-        ensure_y_numeric=False,
-        max_num_samples=InferenceConfig.MAX_NUMBER_OF_SAMPLES,
-        max_num_features=InferenceConfig.MAX_NUMBER_OF_FEATURES,
-        ignore_pretraining_limits=False,
-    )
-
-    if feature_names_in is not None:
-        cls.feature_names_in_ = feature_names_in
-    cls.n_features_in_ = n_features_in
-
-    if not cls.differentiable_input:
-        _, counts = np.unique(y, return_counts=True)
-        cls.class_counts_ = counts
-        cls.label_encoder_ = LabelEncoder()
-        y = cls.label_encoder_.fit_transform(y)
-        cls.classes_ = cls.label_encoder_.classes_
-        cls.n_classes_ = len(cls.classes_)
-    else:
-        cls.label_encoder_ = None
-        if not hasattr(cls, "n_classes_"):
-            cls.n_classes_ = int(torch.max(torch.tensor(y)).item()) + 1
-        cls.classes_ = torch.arange(cls.n_classes_)
-
-    cls.inferred_categorical_indices_ = infer_categorical_features(
-        X=X,
-        provided=categorical_idx,
-        min_samples_for_inference=InferenceConfig.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
-        max_unique_for_category=InferenceConfig.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
-        min_unique_for_numerical=InferenceConfig.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
-    )
-    return (
-        fix_dtypes(X, cat_indices=cls.inferred_categorical_indices_),
-        cls.inferred_categorical_indices_,
-        request.param["ground_truth"],
-    )
-
-
-# --- Actual test ---
-# This is a test for the OrderPreservingColumnTransformer, which is not used currently
-# But might be used in the future, therefore I'll leave it in.
-@pytest.mark.skip
-def test_process_text_na_dataframe(prepared_tabpfn_data):
-    X, categorical_idx, ground_truth = prepared_tabpfn_data  # use the fixture
-
-    ord_encoder = get_ordinal_encoder()
-    X_out = process_text_na_dataframe(
-        X,
-        placeholder=NA_PLACEHOLDER,
-        ord_encoder=ord_encoder,
-        fit_encoder=True,
-    )
-
-    assert X_out.shape[0] == ground_truth.shape[0]
-    assert X_out.shape[1] == ground_truth.shape[1]
-
-    for col_name in X.columns:
-        # col_name should already be a numeric index but using get_loc for safety
-        col_idx = X.columns.get_loc(col_name)
-        original_col = X[col_name].to_numpy()
-        output_col = X_out[:, col_idx]
-        gt_col = ground_truth[:, col_idx]
-        if col_idx not in categorical_idx:
-            # For numeric columns, values should be preserved (within float tolerance).
-            # NaNs should also be in the same positions.
-            np.testing.assert_allclose(
-                output_col,
-                original_col,
-                equal_nan=True,
-                rtol=1e-5,
-            )
-        else:
-            # OrdinalEncoder does not guarante that element order is preserved:
-
-            # First, check if np.nan are correctly positioned
-            # ! use np.isnan on outputcol -> must be numerical
-            # ! use pd.isna on original col -> can be any type
-            np.testing.assert_array_equal(np.isnan(output_col), pd.isna(original_col))
-            # Second, check if there are as many unique non-nan values, as expected
-            # e.g.: ["high", "mid", "low"] -> [0,2,1] or [2,1,0],...
-            assert len(np.unique(output_col[~pd.isna(output_col)])) == len(
-                np.unique(gt_col[~pd.isna(gt_col)])
-            )
-
-
-def test_balance_probas_by_class_counts():
-    """Test balancing probabilities by class counts."""
-    probas = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
-    class_counts = np.array([1, 2])
-
-    balanced = balance_probas_by_class_counts(probas, class_counts)
-
-    # Check that each row sums to one
-    sums = balanced.sum(dim=-1)
-    assert torch.allclose(sums, torch.ones(3), rtol=1e-5, atol=1e-5)
-
-    expected_balanced = torch.tensor([[1 / 3, 2 / 3], [0.75, 0.25], [2 / 3, 1 / 3]])
-    assert torch.allclose(balanced, expected_balanced, rtol=1e-4, atol=1e-4)
+    threads = []
+    for _ in range(10):
+        t = threading.Thread(target=get_memory)
+        threads.append(t)
+        t.start()
+    for t in threads:
+        t.join()
+    psutil_result = psutil.virtual_memory().total / 1e9
+    assert all(result == psutil_result for result in results)
diff --git a/tests/utils.py b/tests/utils.py
index 60f7484..7206ddb 100644
--- a/tests/utils.py
+++ b/tests/utils.py
@@ -1,31 +1,8 @@
 from __future__ import annotations
 
-import os
-
 import torch
 
 
-def get_pytest_devices() -> list[str]:
-    exclude_devices = {
-        d.strip()
-        for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",")
-        if d.strip()
-    }
-
-    devices = []
-    if "cpu" not in exclude_devices:
-        devices.append("cpu")
-    if torch.cuda.is_available() and "cuda" not in exclude_devices:
-        devices.append("cuda")
-    if torch.backends.mps.is_available() and "mps" not in exclude_devices:
-        devices.append("mps")
-
-    if len(devices) == 0:
-        raise RuntimeError("No devices available for testing.")
-
-    return devices
-
-
 def check_cpu_float16_support() -> bool:
     """Checks if CPU float16 operations are supported by attempting a minimal operation.
     Returns True if supported, False otherwise.
