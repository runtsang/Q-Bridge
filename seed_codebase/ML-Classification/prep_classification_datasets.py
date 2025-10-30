# prep_classification_datasets.py
import os
import numpy as np
from typing import List
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits, make_classification
)

OUT_DIR = "datasets_cls"

def _save_npz(path, X, y, feature_names, target_name, dataset_name, class_names=None, description=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # sanitize types to avoid object dtype
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    feature_names = np.asarray([str(f) for f in feature_names], dtype="U")
    target_name = np.asarray(str(target_name), dtype="U")
    dataset_name = np.asarray(str(dataset_name), dtype="U")
    description = np.asarray(str(description), dtype="U")

    # optional class info
    if class_names is None:
        class_names = np.asarray([], dtype="U")
        n_classes = np.unique(y).shape[0]
    else:
        class_names = np.asarray([str(c) for c in class_names], dtype="U")
        n_classes = len(class_names)

    np.savez_compressed(
        path,
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        dataset_name=dataset_name,
        description=description,
        n_samples=np.array(X.shape[0]),
        n_features=np.array(X.shape[1]),
        n_classes=np.array(n_classes),
        class_names=class_names,
    )
    print(f"Saved: {path} (X: {X.shape}, y: {y.shape})")


def iris_npz():
    d = load_iris()
    _save_npz(
        os.path.join(OUT_DIR, "iris_cls.npz"),
        d.data, d.target,
        d.feature_names,
        [str(c) for c in d.target_names],
        "Iris-3class",
        "Iris multiclass classification (3 classes)."
    )

def wine_npz():
    d = load_wine()
    _save_npz(
        os.path.join(OUT_DIR, "wine_cls.npz"),
        d.data, d.target,
        d.feature_names,
        [str(c) for c in d.target_names],
        "Wine-3class",
        "Wine quality classification (3 classes)."
    )

def breast_cancer_npz():
    d = load_breast_cancer()
    _save_npz(
        os.path.join(OUT_DIR, "breast_cancer_cls.npz"),
        d.data, d.target,
        list(d.feature_names),
        ["malignant", "benign"],  # sklearn orders 0/1 like this dataset
        "BreastCancer-binary",
        "Breast cancer binary classification."
    )

def digits_npz():
    d = load_digits()  # 1797 samples, 64 features, 10 classes
    _save_npz(
        os.path.join(OUT_DIR, "digits_cls.npz"),
        d.data, d.target,
        [f"p{i}" for i in range(d.data.shape[1])],
        [str(i) for i in range(10)],
        "Digits-10class",
        "Handwritten digits (0-9). Recommend PCA→4-8 comps for QML."
    )

def synthetic_npz(n=1000, features=6, classes=3, seed=7):
    X, y = make_classification(
        n_samples=n, n_features=features, n_informative=max(2, classes),
        n_redundant=0, n_repeated=0, n_classes=classes,
        n_clusters_per_class=1, flip_y=0.03, class_sep=1.2, random_state=seed
    )
    _save_npz(
        os.path.join(OUT_DIR, f"synthetic_{classes}c_{features}f.npz"),
        X, y,
        [f"x{i+1}" for i in range(features)],
        [f"class_{i}" for i in range(classes)],
        f"Synth-{classes}c-{features}f",
        "Synthetic classification via make_classification."
    )

if __name__ == "__main__":
    iris_npz()
    wine_npz()
    breast_cancer_npz()
    digits_npz()
    synthetic_npz(n=900, features=6, classes=3, seed=13)
