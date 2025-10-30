"""Cross-validation utilities for the PyTorch MLP regressor."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ml_model import build_mlp


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load (X, y) and metadata from an .npz created by prep_datasets.py."""
    with np.load(path, allow_pickle=False) as z:
        X = z["X"]
        y = z["y"]
        meta = {
            "feature_names": z["feature_names"].tolist(),
            "target_name": str(z["target_name"]),
            "dataset_name": str(z["dataset_name"]),
            "description": str(z.get("description", "")),
            "n_samples": int(z["n_samples"]),
            "n_features": int(z["n_features"]),
        }
    return X, y, meta


def _maybe_cast_pca_n(n_raw):
    if n_raw is None:
        return None
    try:
        n_float = float(n_raw)
    except Exception:
        return n_raw
    if abs(n_float - int(n_float)) < 1e-9:
        return int(n_float)
    return n_float


def _fit_transform_scalers(X_tr, y_tr):
    xsc = StandardScaler().fit(X_tr)
    ysc = StandardScaler().fit(y_tr.reshape(-1, 1))
    return xsc, ysc


def _maybe_fit_pca(X_tr_s, use_pca, n_components, whiten, seed):
    if not use_pca:
        return None, None
    pca = PCA(n_components=n_components, whiten=whiten, random_state=seed).fit(X_tr_s)
    evr = getattr(pca, "explained_variance_ratio_", None)
    evr_cum = float(np.cumsum(evr)[-1]) if evr is not None else None
    return pca, evr_cum


def evaluate_file(
    data_path: str,
    *,
    hidden_dims: Tuple[int, int] = (64, 32),
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 200,
    batch_size: int = 32,
    kfold: int = 5,
    random_state: int = 42,
    use_pca: bool = False,
    pca_n_components=None,
    pca_whiten: bool = False,
    save_dir: str | None = None,
    plot_hist: bool = False,
    limit_samples: int | None = None,
    strict_fold_preprocess: bool = False,
    target_on_original_scale: bool = False,
    verbose_training: bool = False,
    device: str | None = None,
):
    """Run cross-validation for a dataset and return (avg_mse, residuals)."""

    X, y, meta = load_npz(data_path)

    if limit_samples is not None and limit_samples < X.shape[0]:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], size=limit_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"[{meta['dataset_name']}] Subsampled to {X.shape[0]} rows for faster validation.")

    if device is not None:
        import torch

        torch_device = torch.device(device)
    else:
        torch_device = None

    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    mses = []
    residuals = []

    if not strict_fold_preprocess:
        Xs = StandardScaler().fit_transform(X)
        ysc_full = StandardScaler().fit(y.reshape(-1, 1))
        yn_s = ysc_full.transform(y.reshape(-1, 1)).ravel()

        if use_pca:
            pca = PCA(n_components=pca_n_components, whiten=pca_whiten, random_state=random_state).fit(Xs)
            Xw = pca.transform(Xs)
            evr = getattr(pca, "explained_variance_ratio_", None)
            evr_cum = float(np.cumsum(evr)[-1]) if evr is not None else None
        else:
            pca, evr_cum = None, None
            Xw = Xs

        n_features_final = Xw.shape[1]
        print(f"[{meta['dataset_name']}] Features → {n_features_final} | use_pca={use_pca}, EVR={evr_cum}")

        for fold, (tr, te) in enumerate(kf.split(Xw), 1):
            mlp = build_mlp(
                n_features_final,
                hidden_layer_sizes=hidden_dims,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                seed=random_state + fold,
                device=torch_device,
            )
            mlp.fit(Xw[tr], yn_s[tr], verbose=verbose_training)
            yhat_s = mlp.predict(Xw[te])

            if target_on_original_scale:
                yhat = ysc_full.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
                fold_resid = y[te] - yhat
            else:
                fold_resid = yn_s[te] - yhat_s

            residuals.append(fold_resid)
            mses.append(float(np.mean(fold_resid**2)))
            print(f"  Fold {fold}/{kfold} MSE: {mses[-1]:.4f}")

    else:
        print(f"[{meta['dataset_name']}] Strict per-fold preprocess enabled.")
        evr_cum = None
        for fold, (tr, te) in enumerate(kf.split(X), 1):
            xsc, ysc = _fit_transform_scalers(X[tr], y[tr])
            X_tr_s = xsc.transform(X[tr])
            X_te_s = xsc.transform(X[te])
            y_tr_s = ysc.transform(y[tr].reshape(-1, 1)).ravel()
            y_te_s = ysc.transform(y[te].reshape(-1, 1)).ravel()

            if use_pca:
                pca, _ = _maybe_fit_pca(X_tr_s, True, pca_n_components, pca_whiten, random_state)
                X_tr_w = pca.transform(X_tr_s)
                X_te_w = pca.transform(X_te_s)
            else:
                X_tr_w, X_te_w = X_tr_s, X_te_s

            n_features_final = X_tr_w.shape[1]
            mlp = build_mlp(
                n_features_final,
                hidden_layer_sizes=hidden_dims,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                seed=random_state + fold,
                device=torch_device,
            )
            mlp.fit(X_tr_w, y_tr_s, verbose=verbose_training)
            yhat_s = mlp.predict(X_te_w)

            if target_on_original_scale:
                yhat = ysc.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
                fold_resid = y[te] - yhat
            else:
                fold_resid = y_te_s - yhat_s

            residuals.append(fold_resid)
            mses.append(float(np.mean(fold_resid**2)))
            print(f"  Fold {fold}/{kfold} MSE: {mses[-1]:.4f}")

    avg_mse = float(np.mean(mses))
    std_mse = float(np.std(mses))
    residuals = np.concatenate(residuals, axis=0)

    print(f"→ Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"→ Error distribution: {residuals.shape[0]} residuals (mean={residuals.mean():.4f}, std={residuals.std():.4f})")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(data_path))[0]
        out_json = os.path.join(save_dir, f"{base}_results_{hidden_dims}.json")
        with open(out_json, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": meta,
                    "settings": {
                        "hidden_dims": list(hidden_dims),
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "kfold": kfold,
                        "strict_fold_preprocess": strict_fold_preprocess,
                        "use_pca": use_pca,
                        "pca_n_components": None if pca_n_components is None else float(_maybe_cast_pca_n(pca_n_components)),
                        "pca_whiten": pca_whiten,
                        "n_features_final": int(n_features_final),
                        "explained_variance_ratio_cum": None if "evr_cum" not in locals() else evr_cum,
                        "limit_samples": limit_samples,
                        "target_on_original_scale": target_on_original_scale,
                    },
                    "metrics": {
                        "avg_mse": avg_mse,
                        "std_mse": std_mse,
                        "residual_mean": float(residuals.mean()),
                        "residual_std": float(residuals.std()),
                        "num_residuals": int(residuals.shape[0]),
                    },
                },
                f,
                indent=2,
            )
        print(f"Saved metrics: {out_json}")

        out_resid = os.path.join(save_dir, f"{base}_residuals.npy")
        np.save(out_resid, residuals)
        print(f"Saved residuals: {out_resid}")

        out_png = os.path.join(save_dir, f"{base}_residual_hist.png")
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=30, alpha=0.9)
        plt.xlabel("Residual" + (" (original scale)" if target_on_original_scale else " (standardized)"))
        plt.ylabel("Count")
        plt.title(f"Residual Histogram: {meta['dataset_name']}")
        plt.tight_layout()
        if plot_hist:
            plt.savefig(out_png, dpi=120)
            print(f"Saved residual histogram: {out_png}")
        plt.close()

    return avg_mse, residuals


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the PyTorch MLP regressor on prepared datasets.")
    ap.add_argument("--data", nargs="+", required=True, help="One or more .npz dataset paths.")
    ap.add_argument("--hidden-dims", nargs=2, type=int, metavar=("H1", "H2"), default=(64, 32))
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization strength.")
    ap.add_argument("--epochs", type=int, default=1, help="Training epochs per fold.")
    ap.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for training.")
    ap.add_argument("--kfold", type=int, default=5, help="Number of cross-validation folds.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed base for reproducibility.")
    ap.add_argument("--use-pca", action="store_true", help="Enable PCA preprocessing.")
    ap.add_argument("--pca-n-components", default=None, type=_maybe_cast_pca_n, help="PCA components (int or fraction).")
    ap.add_argument("--pca-whiten", action="store_true", help="Whiten PCA outputs.")
    ap.add_argument("--save-dir", default=None, help="Directory to save metrics/residuals (default: no save).")
    ap.add_argument("--plot-hist", action="store_true", help="If saving, also write residual histogram PNG.")
    ap.add_argument("--limit-samples", type=int, default=None, help="Subsample rows for speed.")
    ap.add_argument("--strict-fold-preprocess", action="store_true", help="Fit scaler/PCA inside each CV fold.")
    ap.add_argument("--target-on-original-scale", action="store_true", help="Report residuals/MSE on original scale.")
    ap.add_argument("--verbose-training", action="store_true", help="Log training loss during fitting.")
    ap.add_argument("--device", default=None, help="Torch device override (e.g., 'cpu' or 'cuda').")
    args = ap.parse_args()

    for path in args.data:
        print(f"\n=== Validating {path} ===")
        evaluate_file(
            data_path=path,
            hidden_dims=tuple(args.hidden_dims),
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            kfold=args.kfold,
            random_state=args.seed,
            use_pca=args.use_pca,
            pca_n_components=args.pca_n_components,
            pca_whiten=args.pca_whiten,
            save_dir=args.save_dir,
            plot_hist=args.plot_hist,
            limit_samples=args.limit_samples,
            strict_fold_preprocess=args.strict_fold_preprocess,
            target_on_original_scale=args.target_on_original_scale,
            verbose_training=args.verbose_training,
            device=args.device,
        )


if __name__ == "__main__":
    main()
