# validate.py
import os, json, argparse
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from quantum_model import build_vqr  # <-- import the model module

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
    # allow ints or floats (0,1] for variance
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
    optimizer: str = "COBYLA",
    num_reps: int = 3,
    kfold: int = 5,
    random_state: int = 42,
    use_pca: bool = False,
    pca_n_components=None,      # int or float(0,1]
    pca_whiten: bool = False,
    save_dir: str | None = None,
    plot_hist: bool = False,
    limit_samples: int | None = None,
    strict_fold_preprocess: bool = False,
    target_on_original_scale: bool = False,
):
    X, y, meta = load_npz(data_path)

    # optional subsampling for speed
    if limit_samples is not None and limit_samples < X.shape[0]:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], size=limit_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"[{meta['dataset_name']}] Subsampled to {X.shape[0]} rows for faster validation.")

    # Global preprocess (fast, slightly leaky) OR do it per-fold (strict)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    mses = []
    residuals = []

    if not strict_fold_preprocess:
        # Fast: fit scalers and (optional) PCA on FULL X
        Xs = StandardScaler().fit_transform(X)
        yn_s = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

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
            vqr = build_vqr(n_features_final, num_reps=num_reps, optimizer_name=optimizer)
            vqr.fit(Xw[tr], yn_s[tr])
            yhat_s = vqr.predict(Xw[te])

            if target_on_original_scale:
                # scale back to original units for residuals/MSE
                # we need a scaler on y; re-fit so we can inverse (approx)
                ysc_full = StandardScaler().fit(y.reshape(-1, 1))
                yhat = ysc_full.inverse_transform(yhat_s.reshape(-1, 1)).ravel()
                fold_resid = y[te] - yhat
            else:
                fold_resid = yn_s[te] - yhat_s

            residuals.append(fold_resid)
            mses.append(float(np.mean(fold_resid**2)))
            print(f"  Fold {fold}/{kfold} MSE: {mses[-1]:.4f}")

    else:
        # Strict: fit scalers and PCA *inside* each fold using only training split
        print(f"[{meta['dataset_name']}] Strict per-fold preprocess enabled.")
        evr_cum = None  # varies per fold if PCA is used
        for fold, (tr, te) in enumerate(kf.split(X), 1):
            xsc, ysc = _fit_transform_scalers(X[tr], y[tr])
            X_tr_s = xsc.transform(X[tr]); X_te_s = xsc.transform(X[te])
            y_tr_s = ysc.transform(y[tr].reshape(-1, 1)).ravel()
            y_te_s = ysc.transform(y[te].reshape(-1, 1)).ravel()

            if use_pca:
                pca, _ = _maybe_fit_pca(X_tr_s, True, pca_n_components, pca_whiten, random_state)
                X_tr_w = pca.transform(X_tr_s); X_te_w = pca.transform(X_te_s)
            else:
                X_tr_w, X_te_w = X_tr_s, X_te_s

            n_features_final = X_tr_w.shape[1]
            vqr = build_vqr(n_features_final, num_reps=num_reps, optimizer_name=optimizer)
            vqr.fit(X_tr_w, y_tr_s)
            yhat_s = vqr.predict(X_te_w)

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

    # optional saves
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(data_path))[0]
        # Save metrics JSON
        out_json = os.path.join(save_dir, f"{base}_results.json")
        with open(out_json, "w") as f:
            json.dump(
                {
                    "dataset": meta,
                    "settings": {
                        "optimizer": optimizer,
                        "num_reps": num_reps,
                        "kfold": kfold,
                        "strict_fold_preprocess": strict_fold_preprocess,
                        "use_pca": use_pca,
                        "pca_n_components": None if pca_n_components is None else float(_maybe_cast_pca_n(pca_n_components)),
                        "pca_whiten": pca_whiten,
                        "n_features_final": int(n_features_final),
                        "explained_variance_ratio_cum": None if 'evr_cum' not in locals() else evr_cum,
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

        # Save residuals array
        out_resid = os.path.join(save_dir, f"{base}_residuals.npy")
        np.save(out_resid, residuals)
        print(f"Saved residuals: {out_resid}")

        # Optional: save histogram figure
        out_png = os.path.join(save_dir, f"{base}_residual_hist.png")
        plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=30, alpha=0.9)
        plt.xlabel("Residual" + (" (original scale)" if target_on_original_scale else " (standardized)") )
        plt.ylabel("Count")
        plt.title(f"Residual Histogram: {meta['dataset_name']}")
        plt.tight_layout()
        if plot_hist:
            plt.savefig(out_png, dpi=120)
            print(f"Saved residual histogram: {out_png}")
        plt.close()

    # return the two required outputs programmatically
    return avg_mse, residuals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="One or more .npz dataset paths.")
    ap.add_argument("--optimizer", default="COBYLA")
    ap.add_argument("--num-reps", type=int, default=3)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # PCA options
    ap.add_argument("--use-pca", action="store_true")
    ap.add_argument("--pca-n-components", default=None, type=_maybe_cast_pca_n,
                    help="int or fraction in (0,1]; e.g., 4 or 0.95")
    ap.add_argument("--pca-whiten", action="store_true")
    # outputs / speed
    ap.add_argument("--save-dir", default=None, help="Directory to save metrics/residuals (default: no save).")
    ap.add_argument("--plot-hist", action="store_true", help="If saving, also write residual histogram PNG.")
    ap.add_argument("--limit-samples", type=int, default=None, help="Subsample rows for speed.")
    ap.add_argument("--strict-fold-preprocess", action="store_true",
                    help="Fit scaler/PCA inside each CV fold (no leakage, slower).")
    ap.add_argument("--target-on-original-scale", action="store_true",
                    help="Compute residuals/MSE on original target scale (default: standardized).")
    args = ap.parse_args()

    for path in args.data:
        print(f"\n=== Validating {path} ===")
        evaluate_file(
            data_path=path,
            optimizer=args.optimizer,
            num_reps=args.num_reps,
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
        )

if __name__ == "__main__":
    main()













