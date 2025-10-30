# Classical Regression with PyTorch

This directory mirrors the structure of `QML-Regression` but implements a
classical baseline using a three-layer **Multilayer Perceptron (MLP)** in
PyTorch.  The workflow consists of the same three entry points:

1. `prep_datasets.py` – build and store regression datasets in the `datasets/`
   folder (identical to the quantum version).
2. `ml_model.py` – defines `build_mlp`, a configurable three-layer PyTorch
   regressor (`Linear → ReLU → Linear → ReLU → Linear`).
3. `validate.py` – performs K-fold cross-validation with the same preprocessing
   choices (standardisation, optional PCA) used in the quantum pipeline and
   reports MSE/residual statistics.

The goal is to offer a like-for-like classical counterpart that can be trained
on the same datasets and produce comparable mean squared errors.

## Requirements

- Python 3.10+
- `torch`
- `scikit-learn`
- `numpy`
- `matplotlib`

## Quickstart

1. **Create datasets** (only needs to be done once; identical to the QML
   project):

   ```bash
   python prep_datasets.py
   ```

   This writes `.npz` bundles to `datasets/` with fields `X`, `y`, feature
   metadata, etc.

2. **Validate the MLP** on one or more datasets:

   ```bash
   python validate.py --data datasets/iris_regression.npz
   ```

   Useful options:

   - `--hidden-dims 128 64` – change hidden layer widths.
   - `--epochs 300 --batch-size 16` – adjust training loop.
   - `--use-pca --pca-n-components 0.95` – apply PCA before training.
   - `--save-dir outputs --plot-hist` – persist metrics, residuals, and
     histograms.

The script prints per-fold MSEs along with the aggregate average/standard
deviation and residual distribution, matching the outputs from the quantum
validation utility.
