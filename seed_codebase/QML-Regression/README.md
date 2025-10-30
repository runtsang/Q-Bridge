# Quantum Regression with Qiskit Machine Learning

This project implements a compact but flexible pipeline to run **Variational Quantum Regression (VQR)** on classical datasets using [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/).

It is organized into three scripts:

1. [`prep_datasets.py`](#1-prep_datasetspy) – prepares and saves datasets in a uniform `.npz` format.  
2. [`quantum_model.py`](#2-quantum_modelpy) – builds a configurable Variational Quantum Regressor (VQR).  
3. [`validate.py`](#3-validatepy) – trains and evaluates the model with cross-validation, producing metrics and error distributions.

---

## Requirements

- Python 3.10+
- Packages:
  - `qiskit`
  - `qiskit-machine-learning`
  - `qiskit-algorithms`
  - `scikit-learn`
  - `numpy`
  - `matplotlib`

Install (example):

```bash
pip install qiskit qiskit-machine-learning qiskit-algorithms scikit-learn numpy matplotlib
```

> By default we use the **StatevectorEstimator** (noiseless simulator). If you switch to hardware/noisy backends, expect longer runtimes and stochasticity.

---

## Project Layout

```
.
├── prep_datasets.py       # Build & save datasets to datasets/*.npz
├── quantum_model.py       # VQR builder (feature map, ansatz, optimizer, estimator)
├── validate.py            # Cross-validation + metrics & residuals
└── datasets/              # Output folder created by prep_datasets.py
```

---

## Quickstart

1) **Create datasets**

```bash
python prep_datasets.py
```

This generates five `.npz` files under `datasets/`:

- `iris_regression.npz` – Iris, predict petal length from the other 3 features.  
- `diabetes.npz` – 10-feature regression.  
- `california_housing.npz` – 8-feature house pricing regression.  
- `synthetic4.npz` – 4-feature synthetic linear regression.  
- `synthetic6.npz` – 6-feature synthetic regression via `make_regression`.

2) **Validate a model** (examples below) using `validate.py`.

---

## 1) `prep_datasets.py`

### Purpose
Standardizes dataset preparation and writes a consistent `.npz` bundle per dataset.

### Saved `.npz` contents
- `X`: feature matrix `(n_samples, n_features)`  
- `y`: target vector `(n_samples,)`  
- `feature_names`: list of feature names  
- `target_name`: target variable name  
- `dataset_name`: a short identifier  
- `description`: free-text description  
- `n_samples`, `n_features`: dataset sizes

### Run
```bash
python prep_datasets.py
```
Outputs are placed in `datasets/` automatically.

---

## 2) `quantum_model.py`

### Purpose
Defines a function to **build a VQR** for a given feature dimension.

### Components
- **Feature map:** `ZZFeatureMap` (dimension = number of features / qubits).  
- **Ansatz:** `RealAmplitudes` with linear entanglement (depth controlled by `num_reps`).  
- **Optimizers:** choose from `COBYLA` (default), `SPSA`, `L_BFGS_B`, `ADAM`, `POWELL`, `CRS`, etc.  
- **Estimator:** `StatevectorEstimator` (noiseless simulation).

### Public API
```python
from quantum_model import build_vqr

vqr = build_vqr(n_features=4, num_reps=3, optimizer_name="COBYLA")
```
`validate.py` imports this function to construct the model per fold.

---

## 3) `validate.py`

### Purpose
Loads `.npz` datasets, applies preprocessing, runs **K-fold cross-validation**, and produces **metrics** and **error distributions**.

### Key features
- **Preprocessing**
  - Standardization (`StandardScaler`) on features (and on target for loss in standardized units).
  - Optional **PCA**: `--use-pca` with `--pca-n-components` (int or fraction like `0.95`) and `--pca-whiten`.
- **Cross-validation**
  - `--kfold` splits the data; each fold trains on `k-1` partitions and validates on 1.
  - Two regimes:
    - *Fast (default):* Fit scalers/PCA on the full dataset once (slight information leakage but faster).
    - *Strict:* `--strict-fold-preprocess` fits scalers/PCA **inside each fold** using only training data (no leakage, slower).
- **Residual analysis**
  - Residuals = `(actual − predicted)`. Collected across all folds.
  - Saved as `.npy` and optionally plotted as a histogram (`--plot-hist`).

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path(s) to one or more `.npz` dataset(s). | **Required** |
| `--optimizer` | Optimizer to use (`COBYLA`, `SPSA`, `L_BFGS_B`, etc.). | COBYLA |
| `--num-reps` | Circuit depth (repetitions for feature map + ansatz). | 3 |
| `--kfold` | Number of cross-validation folds. | 5 |
| `--seed` | Random seed for reproducibility. | 42 |
| `--use-pca` | Enable PCA preprocessing. | Off |
| `--pca-n-components` | PCA components (int or fraction of variance in `(0,1]`). | None |
| `--pca-whiten` | Whiten PCA outputs. | Off |
| `--save-dir` | Directory to save outputs. | None |
| `--plot-hist` | Save residual histogram as `.png`. | Off |
| `--limit-samples` | Subsample dataset for speed. | None |
| `--strict-fold-preprocess` | Fit scaler/PCA separately in each fold (no leakage). | Off |
| `--target-on-original-scale` | Compute residuals/MSE in original target units. | Off |

---

## Outputs & How to Interpret Them

When `--save-dir` is provided, the script writes to that directory:

- `*_results.json` – dataset info, settings, and **metrics** (see below).  
- `*_residuals.npy` – raw residual array (one value per validation sample).  
- `*_residual_hist.png` – residual histogram (created only if `--plot-hist`).  

### The metrics JSON (`*_results.json`)

```json
{
  "dataset": {
    "feature_names": [...],
    "target_name": "target",
    "dataset_name": "Diabetes-10f",
    "description": "Standard diabetes regression (10 features -> target).",
    "n_samples": 442,
    "n_features": 10
  },
  "settings": {
    "optimizer": "COBYLA",
    "num_reps": 3,
    "kfold": 5,
    "strict_fold_preprocess": false,
    "use_pca": true,
    "pca_n_components": 4,
    "pca_whiten": false,
    "n_features_final": 4,
    "explained_variance_ratio_cum": 0.95,
    "limit_samples": null,
    "target_on_original_scale": false
  },
  "metrics": {
    "avg_mse": 0.042,
    "std_mse": 0.011,
    "residual_mean": -0.001,
    "residual_std": 0.204,
    "num_residuals": 442
  }
}
```

#### What each metric means

- **`avg_mse`** – The **average** mean squared error across the `k` validation folds.  
  - Lower is better.  
  - If `target_on_original_scale` is **false** (default), the target is standardized, so `avg_mse` is in **“standard deviation units squared”**.  
    - Rule of thumb: `avg_mse = 1.0` ⇒ the prediction error is about one standard deviation of the original target.  
    - `avg_mse = 0.25` ⇒ RMSE ≈ `0.5` std-devs (quite good).  
  - If `target_on_original_scale` is **true**, the MSE is in the **original target units** (e.g., dollars, mm, etc.).

- **`std_mse`** – The **standard deviation** of per-fold MSE.  
  - Lower means performance is more consistent across folds.

- **`residual_mean`** – The mean of all residuals `(y_true − y_pred)` gathered from the validation splits.  
  - Should be close to 0 for an **unbiased** model.  
  - On standardized target scale, values within `±0.05` are typically fine.

- **`residual_std`** – The standard deviation of residuals.  
  - Smaller is better; it measures the spread of prediction errors.  
  - On standardized scale, it is comparable across datasets; on original scale, it inherits the target’s units.

- **`num_residuals`** – Number of residuals (usually equals the dataset size if every sample appears once in a validation fold).

#### Residuals array (`*_residuals.npy`)

- A 1D NumPy array of residuals for **all validation predictions across folds**.  
- Use this for custom analysis:
  ```python
  import numpy as np
  res = np.load("val_outputs/diabetes_residuals.npy")
  global_mse = np.mean(res**2)
  ```
- The histogram (`*_residual_hist.png`) visualizes error distribution (e.g., symmetry, skewness, heavy tails).

> **Standardized vs Original scale:**  
> By default, metrics are computed on the **standardized** target (mean 0, std 1). This makes numbers comparable across datasets. If you pass `--target-on-original-scale`, residuals/MSE will be in the original units and **not** directly comparable across datasets.

---

## Example Commands

**A. Diabetes + PCA (4 comps → 4 qubits), save metrics & histogram**
```bash
python validate.py --data datasets/diabetes.npz \
  --use-pca --pca-n-components 4 \
  --save-dir val_outputs --plot-hist
```

**B. California Housing (large) with PCA + fewer folds + (optional) subsample for speed**
```bash
python validate.py --data datasets/california_housing.npz \
  --use-pca --pca-n-components 4 --kfold 3 --limit-samples 4000 \
  --save-dir val_outputs --plot-hist
```

**C. Iris (no PCA) with strict per-fold preprocessing (no leakage)**
```bash
python validate.py --data datasets/iris_regression.npz \
  --strict-fold-preprocess \
  --save-dir val_outputs
```

**D. Run many datasets at once**
```bash
python validate.py --data datasets/*.npz --use-pca --pca-n-components 4 \
  --save-dir val_outputs
```

---

## Why `validate.py` may run slower than a single demo script

- **K-fold CV** trains the model **k** times (once per fold).  
- **Strict mode** (`--strict-fold-preprocess`) refits scalers/PCA every fold for correctness.  
- **Larger datasets** (e.g., California Housing) and deeper circuits (`num_reps`) increase runtime.

**Tips to speed up:**
- Use `--kfold 2` while iterating; raise later for final results.  
- Use `--limit-samples` during prototyping.  
- Reduce `--num-reps`, or choose a faster optimizer.  
- Apply PCA to reduce the feature dimension → fewer qubits.

---

## Troubleshooting

- **“Input data has incorrect shape…”**  
  This usually means the **qubit count** doesn’t match the **feature dimension**. Here, `validate.py` always uses the post-PCA dimension (`n_features_final`) to build `VQR`, so shapes should match unless the files or code were modified independently.

- **Slow runs on California Housing**  
  Use PCA, fewer folds (`--kfold 3`), or `--limit-samples` for quick checks.

- **Different metric scales**  
  Pass `--target-on-original-scale` if you need error metrics in physical units instead of standardized units.

---

## License

This project is provided for educational/research purposes. Use at your own discretion.
