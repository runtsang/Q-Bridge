"""Cross-validation utilities for the PyTorch MLP classifier."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from ml_model_cls import build_mlp_classifier


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load (X, y) and metadata from a .npz dataset."""
    try:
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
            if "n_classes" in z:
                meta["n_classes"] = int(z["n_classes"])
            if "class_names" in z:
                meta["class_names"] = [str(c) for c in z["class_names"].tolist()]
    except ValueError as exc:
        if "allow_pickle=False" not in str(exc):
            raise
        with np.load(path, allow_pickle=True) as z:
            X = z["X"]
            y = z["y"]
            meta = {
                "feature_names": [str(f) for f in z.get("feature_names", [])],
                "target_name": str(z.get("target_name", "target")),
                "dataset_name": str(z.get("dataset_name", os.path.basename(path))),
                "description": str(z.get("description", "")),
                "n_samples": int(z.get("n_samples", X.shape[0])),
                "n_features": int(z.get("n_features", X.shape[1])),
            }
            if "n_classes" in z:
                meta["n_classes"] = int(z["n_classes"])
            if "class_names" in z:
                meta["class_names"] = [str(c) for c in z.get("class_names", [])]
        print("[WARN] Loaded dataset with allow_pickle=True; metadata sanitized to strings.")
    return X, y, meta


def _maybe_cast_pca_n(n_raw):
    if n_raw is None:
        return None
    try:
        val = float(n_raw)
    except Exception:
        return n_raw
    if abs(val - int(val)) < 1e-9:
        return int(val)
    return val


def _prepare_labels(y: np.ndarray, class_names_meta: Sequence[str] | None):
    """Ensure labels are integer encoded and return (labels, classes, class_names)."""
    y_arr = np.asarray(y)
    if y_arr.dtype.kind not in "iu":
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y_arr)
        classes = np.arange(len(encoder.classes_))
        class_names = [str(c) for c in encoder.classes_]
        return y_enc, classes, class_names

    y_enc = y_arr.astype(int)
    classes = np.unique(y_enc)
    if class_names_meta:
        class_names = [
            str(class_names_meta[int(cls)]) if int(cls) < len(class_names_meta) else str(cls)
            for cls in classes
        ]
    else:
        class_names = [str(c) for c in classes]
    return y_enc, classes, class_names


def evaluate_file(
    data_path: str,
    *,
    hidden_dims: Tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
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
    limit_samples: int | None = None,
    strict_fold_preprocess: bool = False,
    verbose_training: bool = False,
    device: str | None = None,
    class_weight: str | None = None,
):
    X, y_raw, meta = load_npz(data_path)

    if limit_samples is not None and limit_samples < X.shape[0]:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], size=limit_samples, replace=False)
        X, y_raw = X[idx], y_raw[idx]
        print(f"[{meta['dataset_name']}] Subsampled to {X.shape[0]} rows for faster validation.")

    y, classes, class_names = _prepare_labels(y_raw, meta.get("class_names"))
    n_classes = len(classes)

    if device is not None:
        import torch

        torch_device = torch.device(device)
    else:
        torch_device = None

    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    accs, f1_macros, f1_weights = [], [], []
    cm_total = np.zeros((n_classes, n_classes), dtype=int)
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    if not strict_fold_preprocess:
        Xs = StandardScaler().fit_transform(X)
        if use_pca:
            pca = PCA(
                n_components=pca_n_components, whiten=pca_whiten, random_state=random_state
            ).fit(Xs)
            Xw = pca.transform(Xs)
            evr = getattr(pca, "explained_variance_ratio_", None)
            evr_cum = float(np.cumsum(evr)[-1]) if evr is not None else None
        else:
            pca, evr_cum = None, None
            Xw = Xs

        n_features_final = Xw.shape[1]
        print(
            f"[{meta['dataset_name']}] n_features→{n_features_final} | use_pca={use_pca} EVR={evr_cum}"
        )

        for fold, (tr, te) in enumerate(kf.split(Xw), 1):
            mlp = build_mlp_classifier(
                n_features=n_features_final,
                n_classes=n_classes,
                hidden_layer_sizes=hidden_dims,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                seed=random_state + fold,
                device=torch_device,
            )

            cw = None
            if class_weight == "balanced":
                cw = compute_class_weight(class_weight="balanced", classes=classes, y=y[tr])

            mlp.fit(Xw[tr], y[tr], verbose=verbose_training, class_weights=cw)
            y_pred = mlp.predict(Xw[te])

            accs.append(float(accuracy_score(y[te], y_pred)))
            f1_macros.append(float(f1_score(y[te], y_pred, average="macro")))
            f1_weights.append(float(f1_score(y[te], y_pred, average="weighted")))
            cm_total += confusion_matrix(y[te], y_pred, labels=classes)
            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(y_pred.tolist())

            print(f"  Fold {fold}/{kfold} acc={accs[-1]:.3f} f1_macro={f1_macros[-1]:.3f}")
    else:
        print(f"[{meta['dataset_name']}] Strict per-fold preprocessing enabled.")
        evr_cum = None

        for fold, (tr, te) in enumerate(kf.split(X), 1):
            xsc = StandardScaler().fit(X[tr])
            X_tr_s = xsc.transform(X[tr])
            X_te_s = xsc.transform(X[te])

            if use_pca:
                pca = PCA(
                    n_components=pca_n_components, whiten=pca_whiten, random_state=random_state
                ).fit(X_tr_s)
                X_tr_w = pca.transform(X_tr_s)
                X_te_w = pca.transform(X_te_s)
                n_features_final = X_tr_w.shape[1]
            else:
                X_tr_w, X_te_w = X_tr_s, X_te_s
                n_features_final = X_tr_w.shape[1]

            mlp = build_mlp_classifier(
                n_features=n_features_final,
                n_classes=n_classes,
                hidden_layer_sizes=hidden_dims,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                seed=random_state + fold,
                device=torch_device,
            )

            cw = None
            if class_weight == "balanced":
                cw = compute_class_weight(class_weight="balanced", classes=classes, y=y[tr])

            mlp.fit(X_tr_w, y[tr], verbose=verbose_training, class_weights=cw)
            y_pred = mlp.predict(X_te_w)

            accs.append(float(accuracy_score(y[te], y_pred)))
            f1_macros.append(float(f1_score(y[te], y_pred, average="macro")))
            f1_weights.append(float(f1_score(y[te], y_pred, average="weighted")))
            cm_total += confusion_matrix(y[te], y_pred, labels=classes)
            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(y_pred.tolist())

            print(f"  Fold {fold}/{kfold} acc={accs[-1]:.3f} f1_macro={f1_macros[-1]:.3f}")

    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
    f1m_mean, f1m_std = float(np.mean(f1_macros)), float(np.std(f1_macros))
    f1w_mean, f1w_std = float(np.mean(f1_weights)), float(np.std(f1_weights))

    print(f"→ Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"→ F1 (macro): {f1m_mean:.3f} ± {f1m_std:.3f}")
    print(f"→ F1 (weighted): {f1w_mean:.3f} ± {f1w_std:.3f}")
    print("→ Confusion matrix (aggregated across folds):")
    print(cm_total)

    per_class_acc: Dict[str, float] = {}
    row_sums = cm_total.sum(axis=1).astype(float)
    for idx, cls in enumerate(classes):
        acc_i = (cm_total[idx, idx] / row_sums[idx]) if row_sums[idx] > 0 else 0.0
        if idx < len(class_names):
            name = class_names[idx]
        else:
            name = str(cls)
        per_class_acc[name] = float(acc_i)

    report_text = classification_report(
        y_true_all, y_pred_all, labels=classes, target_names=class_names, digits=4
    )
    print(report_text)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(data_path))[0]
        out_json = os.path.join(save_dir, f"{base}_cls_results_ml.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": meta,
                    "settings": {
                        "hidden_dims": list(hidden_dims),
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "kfold": kfold,
                        "random_state": random_state,
                        "strict_fold_preprocess": strict_fold_preprocess,
                        "use_pca": use_pca,
                        "pca_n_components": None
                        if pca_n_components is None
                        else float(_maybe_cast_pca_n(pca_n_components)),
                        "pca_whiten": pca_whiten,
                        "n_features_final": int(n_features_final),
                        "explained_variance_ratio_cum": evr_cum,
                        "limit_samples": limit_samples,
                        "class_weight": class_weight,
                        "device": str(torch_device) if torch_device is not None else "cpu",
                    },
                    "metrics": {
                        "accuracy_mean": acc_mean,
                        "accuracy_std": acc_std,
                        "f1_macro_mean": f1m_mean,
                        "f1_macro_std": f1m_std,
                        "f1_weighted_mean": f1w_mean,
                        "f1_weighted_std": f1w_std,
                        "per_class_accuracy": per_class_acc,
                        "confusion_matrix": cm_total.tolist(),
                        "class_labels": [int(c) for c in classes],
                        "class_names": class_names,
                        "classification_report": report_text,
                    },
                },
                f,
                indent=2,
            )
        print(f"Saved metrics: {out_json}")

    return acc_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="One or more .npz dataset paths.")
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=(8, 8), help="Hidden layer sizes.")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout probability between hidden layers.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-pca", action="store_true")
    ap.add_argument(
        "--pca-n-components",
        default=None,
        type=_maybe_cast_pca_n,
        help="int or fraction in (0,1]; e.g., 4 or 0.95",
    )
    ap.add_argument("--pca-whiten", action="store_true")
    ap.add_argument("--save-dir", default=None, help="Where to write JSON outputs.")
    ap.add_argument("--limit-samples", type=int, default=None, help="Subsample rows for speed.")
    ap.add_argument(
        "--strict-fold-preprocess",
        action="store_true",
        help="Fit scaler/PCA inside each fold instead of globally.",
    )
    ap.add_argument("--verbose-training", action="store_true")
    ap.add_argument("--device", default=None, help="torch device string, e.g. cuda or cpu.")
    ap.add_argument(
        "--class-weight",
        default=None,
        choices=["balanced"],
        help="Class weighting strategy for CrossEntropyLoss.",
    )

    args = ap.parse_args()

    hidden_dims = tuple(args.hidden_dims)

    for path in args.data:
        print(f"\n=== Validating {path} ===")
        evaluate_file(
            data_path=path,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
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
            limit_samples=args.limit_samples,
            strict_fold_preprocess=args.strict_fold_preprocess,
            verbose_training=args.verbose_training,
            device=args.device,
            class_weight=args.class_weight,
        )


if __name__ == "__main__":
    main()
