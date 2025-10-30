# prep_datasets.py
import os
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing, make_regression

OUT_DIR = "datasets"

def save_npz(path, X, y, feature_names, target_name, dataset_name, description=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        feature_names=np.array(feature_names),
        target_name=np.array(target_name),
        dataset_name=np.array(dataset_name),
        description=np.array(description),
        n_samples=np.array(X.shape[0]),
        n_features=np.array(X.shape[1]),
    )
    print(f"Saved: {path} (X: {X.shape}, y: {y.shape})")

def iris_regression_npz():
    iris = load_iris()
    names = iris.feature_names  # 4 cols
    target_name = "petal length (cm)"
    idx = names.index(target_name)
    X_all = iris.data
    y = X_all[:, idx]                      # target
    X = np.delete(X_all, idx, axis=1)      # 3 features
    save_npz(
        os.path.join(OUT_DIR, "iris_regression.npz"),
        X, y, [n for i, n in enumerate(names) if i != idx],
        target_name, "Iris-Regression-3f",
        "Predict petal length from remaining 3 features."
    )

def diabetes_npz():
    d = load_diabetes()
    X, y = d.data, d.target                # 10 features
    save_npz(
        os.path.join(OUT_DIR, "diabetes.npz"),
        X, y, d.feature_names, "target", "Diabetes-10f",
        "Standard diabetes regression (10 features -> target)."
    )

def california_npz():
    c = fetch_california_housing()         # 8 features (downloads once)
    X, y = c.data, c.target
    save_npz(
        os.path.join(OUT_DIR, "california_housing.npz"),
        X, y, c.feature_names, "MedHouseVal", "CaliforniaHousing-8f",
        "Predict median house value from 8 features."
    )

def synthetic4_npz(n=800, noise=5.0, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    w = np.array([0.8, -1.2, 0.5, 1.7])
    y = X @ w + rng.normal(scale=noise, size=n)
    save_npz(
        os.path.join(OUT_DIR, "synthetic4.npz"),
        X, y, [f"x{i+1}" for i in range(4)], "y", "Synthetic-4f",
        "Synthetic 4-feature linear signal + noise."
    )

def synthetic6_npz(n=1000, noise=8.0, seed=13):
    X, y = make_regression(
        n_samples=n, n_features=6, n_informative=5,
        noise=noise, random_state=seed
    )
    save_npz(
        os.path.join(OUT_DIR, "synthetic6.npz"),
        X, y, [f"x{i+1}" for i in range(6)], "y", "Synthetic-6f",
        "Synthetic 6-feature regression via make_regression."
    )

if __name__ == "__main__":
    iris_regression_npz()
    diabetes_npz()
    california_npz()
    synthetic4_npz()
    synthetic6_npz()
