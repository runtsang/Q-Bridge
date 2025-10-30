# QML Classification (VQC / SamplerQNN) with Qiskit Machine Learning

This folder mirrors your regression pipeline but for **classification**.
It uses your custom circuit variants (`feature_map_variations.py`, `ansatz_variations.py`), and provides:

- `prep_classification_datasets.py` – prepare 5 classification datasets → `.npz`
- `quantum_model_cls.py` – build a quantum classifier (VQC if available; fallback to SamplerQNN)
- `validate_cls.py` – K-fold validation, metrics, confusion matrix → JSON

---

## 1) Prepare datasets

Generates `.npz` files under `datasets_cls/`:

- `iris_cls.npz` (3 classes, 4 features)
- `wine_cls.npz` (3 classes, 13 features)
- `breast_cancer_cls.npz` (binary, 30 features)
- `digits_cls.npz` (10 classes, 64 features)
- `synthetic_3c_6f.npz` (3 classes, 6 features)

```bash
# python prep_classification_datasets.py

Examples: Save JSON to val_cls/: 
1. 'python validate_cls.py --data datasets_cls/iris_cls.npz \ --save-dir val_cls'
2. 'python validate_cls.py --data datasets_cls/wine_cls.npz \ --use-pca --pca-n-components 4 --kfold 3 \ --save-dir val_cls'
3. 'python validate_cls.py --data datasets_cls/digits_cls.npz \ --limit-samples 1500 \ --use-pca --pca-n-components 8 \ --kfold 3 --save-dir val_cls' 