# Classical Classification with PyTorch

This directory mirrors the structure of `QML-Classification` but implements a
purely classical baseline using a three-layer **Multilayer Perceptron (MLP)** in
PyTorch.  It preserves the same workflow and dataset format so results can be
compared like-for-like with the quantum pipeline.

1. `prep_classification_datasets.py` – generates the same `.npz` bundles under
   `datasets_cls/` as the quantum project.
2. `ml_model_cls.py` – provides `build_mlp_classifier`, a configurable PyTorch
   classifier (`Linear → ReLU → (Dropout) → Linear → … → Linear`) with
   cross-entropy training.
3. `validate_cls.py` – runs K-fold cross-validation with the identical
   preprocessing options (standardisation, optional PCA, per-fold pipelines) and
   reports accuracy/F1/confusion matrices.

## Requirements

- Python 3.10+
- `torch`
- `numpy`
- `scikit-learn`

## Quickstart

1. **Create datasets** (only required once; produces the exact inputs used by
   the quantum scripts):

   ```bash
   python prep_classification_datasets.py
   ```

2. **Evaluate the MLP baseline** on a dataset:

   ```bash
   python validate_cls.py --data datasets_cls/iris_cls.npz
   ```

   Useful flags:

   - `--hidden-dims 128 64` – adjust hidden layer widths.
   - `--dropout 0.2` – insert dropout between hidden layers.
   - `--epochs 300 --batch-size 16` – tune the training loop.
   - `--use-pca --pca-n-components 0.95` – apply PCA before the MLP.
   - `--save-dir outputs` – persist metrics/aggregated confusion matrices.

The script prints per-fold accuracy/F1 scores and aggregates them to match the
reporting format of the quantum validation utility.
