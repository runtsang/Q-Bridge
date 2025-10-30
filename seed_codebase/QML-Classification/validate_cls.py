# validate_cls.py
import os, json, argparse
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from quantum_model_cls import build_classifier

def load_npz(path: str):
    """Load (X, y) and metadata from a .npz. Tries safe load first; falls back to allow_pickle=True."""
    def _as_str_list(x):
        return [str(t) for t in x] if isinstance(x, (list, tuple, np.ndarray)) else [str(x)]

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
            # Optional keys if present:
            if "n_classes" in z: meta["n_classes"] = int(z["n_classes"])
            if "class_names" in z: meta["class_names"] = z["class_names"].tolist()
        return X, y, meta
    except ValueError as e:
        if "allow_pickle=False" not in str(e):
            raise

    # Fallback: allow_pickle=True, then sanitize to plain strings/ints
    with np.load(path, allow_pickle=True) as z:
        X = z["X"]
        y = z["y"]
        # force non-object meta fields
        feature_names = _as_str_list(z["feature_names"])
        target_name = str(z.get("target_name", "target"))
        dataset_name = str(z.get("dataset_name", "dataset"))
        description = str(z.get("description", ""))
        n_samples = int(z.get("n_samples", X.shape[0]))
        n_features = int(z.get("n_features", X.shape[1]))
        meta = {
            "feature_names": feature_names,
            "target_name": target_name,
            "dataset_name": dataset_name,
            "description": description,
            "n_samples": n_samples,
            "n_features": n_features,
        }
        if "n_classes" in z: meta["n_classes"] = int(z["n_classes"])
        if "class_names" in z: meta["class_names"] = _as_str_list(z["class_names"])
    print("[WARN] Loaded with allow_pickle=True and sanitized meta to plain strings.")
    return X, y, meta


def _maybe_cast_pca_n(n_raw):
    if n_raw is None: return None
    try:
        f = float(n_raw)
        if abs(f - int(f)) < 1e-9: return int(f)
        return f
    except Exception:
        return n_raw

def _to_one_vs_rest_estimators(
    n_classes: int,
    n_features: int,
    num_reps: int,
    optimizer: str,
    fm_name: str | None,
    an_name: str | None,
):
    # Build one binary classifier per class
    estims = []
    for _ in range(n_classes):
        estims.append(
            build_classifier(
                n_features=n_features,
                num_reps=num_reps,
                optimizer_name=optimizer,
                feature_map_name=fm_name,
                ansatz_name=an_name,
            )
        )
    return estims

def _predict_ovr(estims, X_te: np.ndarray) -> np.ndarray:
    """Combine One-vs-Rest probabilities (or scores) to produce multiclass preds."""
    scores = []
    for clf in estims:
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_te)
            # take probability for class '1' if shape (n,2); or last column otherwise
            if prob.ndim == 2 and prob.shape[1] >= 2:
                scores.append(prob[:, -1])
            else:
                scores.append(prob.ravel())
        elif hasattr(clf, "decision_function"):
            s = clf.decision_function(X_te)
            scores.append(s if s.ndim == 1 else s[:, 0])
        else:
            # fallback to predicted labels as 0/1 score
            scores.append(clf.predict(X_te).astype(float))
    S = np.vstack(scores).T  # (n_samples, n_classes)
    return np.argmax(S, axis=1)

def evaluate_file(
    data_path: str,
    optimizer: str = "COBYLA",
    num_reps: int = 2,
    kfold: int = 5,
    random_state: int = 42,
    use_pca: bool = False,
    pca_n_components=None,
    pca_whiten: bool = False,
    save_dir: str | None = None,
    limit_samples: int | None = None,
    strict_fold_preprocess: bool = False,
    feature_map_name: str | None = None,
    ansatz_name: str | None = None,
):
    X, y, meta = load_npz(data_path)

    if limit_samples is not None and limit_samples < X.shape[0]:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], size=limit_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"[{meta['dataset_name']}] Subsampled to {X.shape[0]} rows for speed.")

    classes = np.unique(y)
    class_names = meta["class_names"]
    n_classes = len(classes)

    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    accs, f1_macros, f1_weights = [], [], []
    cm_total = np.zeros((n_classes, n_classes), dtype=int)

    # -------- preprocess (global or strict per-fold) --------
    if not strict_fold_preprocess:
        Xs = StandardScaler().fit_transform(X)
        if use_pca:
            pca = PCA(n_components=pca_n_components, whiten=pca_whiten, random_state=random_state).fit(Xs)
            Xw = pca.transform(Xs)
            evr = getattr(pca, "explained_variance_ratio_", None)
            evr_cum = float(np.cumsum(evr)[-1]) if evr is not None else None
        else:
            Xw, evr_cum = Xs, None

        n_feat_final = Xw.shape[1]
        print(f"[{meta['dataset_name']}] n_features→{n_feat_final} | use_pca={use_pca} EVR={evr_cum}")

        for fold, (tr, te) in enumerate(kf.split(Xw), 1):
            if n_classes == 2:
                clf = build_classifier(
                    n_features=n_feat_final, num_reps=num_reps, optimizer_name=optimizer,
                    feature_map_name=feature_map_name, ansatz_name=ansatz_name
                )
                clf.fit(Xw[tr], y[tr])
                y_pred = clf.predict(Xw[te])
            else:
                estims = _to_one_vs_rest_estimators(
                    n_classes, n_feat_final, num_reps, optimizer, feature_map_name, ansatz_name
                )
                # fit one-vs-rest
                for c, clf in enumerate(estims):
                    y_bin = (y[tr] == classes[c]).astype(int)
                    clf.fit(Xw[tr], y_bin)
                y_pred = _predict_ovr(estims, Xw[te])

            accs.append(float(accuracy_score(y[te], y_pred)))
            f1_macros.append(float(f1_score(y[te], y_pred, average="macro")))
            f1_weights.append(float(f1_score(y[te], y_pred, average="weighted")))
            cm_total += confusion_matrix(y[te], y_pred, labels=classes)
            print(f"  Fold {fold}/{kfold} acc={accs[-1]:.3f} f1_macro={f1_macros[-1]:.3f}")

    else:
        print(f"[{meta['dataset_name']}] Strict per-fold preprocessing.")
        evr_cum = None
        for fold, (tr, te) in enumerate(kf.split(X), 1):
            xsc = StandardScaler().fit(X[tr])
            X_tr_s, X_te_s = xsc.transform(X[tr]), xsc.transform(X[te])

            if use_pca:
                pca = PCA(n_components=pca_n_components, whiten=pca_whiten, random_state=random_state).fit(X_tr_s)
                X_tr_w, X_te_w = pca.transform(X_tr_s), pca.transform(X_te_s)
            else:
                X_tr_w, X_te_w = X_tr_s, X_te_s

            n_feat_final = X_tr_w.shape[1]

            if n_classes == 2:
                clf = build_classifier(
                    n_features=n_feat_final, num_reps=num_reps, optimizer_name=optimizer,
                    feature_map_name=feature_map_name, ansatz_name=ansatz_name
                )
                clf.fit(X_tr_w, y[tr])
                y_pred = clf.predict(X_te_w)
            else:
                estims = _to_one_vs_rest_estimators(
                    n_classes, n_feat_final, num_reps, optimizer, feature_map_name, ansatz_name
                )
                for c, clf in enumerate(estims):
                    y_bin = (y[tr] == classes[c]).astype(int)
                    clf.fit(X_tr_w, y_bin)
                y_pred = _predict_ovr(estims, X_te_w)

            accs.append(float(accuracy_score(y[te], y_pred)))
            f1_macros.append(float(f1_score(y[te], y_pred, average="macro")))
            f1_weights.append(float(f1_score(y[te], y_pred, average="weighted")))
            cm_total += confusion_matrix(y[te], y_pred, labels=classes)
            print(f"  Fold {fold}/{kfold} acc={accs[-1]:.3f} f1_macro={f1_macros[-1]:.3f}")

    # -------- summarize --------
    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
    f1m_mean, f1m_std = float(np.mean(f1_macros)), float(np.std(f1_macros))
    f1w_mean, f1w_std = float(np.mean(f1_weights)), float(np.std(f1_weights))

    print(f"→ Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"→ F1 (macro): {f1m_mean:.3f} ± {f1m_std:.3f}")
    print(f"→ F1 (weighted): {f1w_mean:.3f} ± {f1w_std:.3f}")
    print("→ Confusion matrix (aggregated across folds):")
    print(cm_total)

    # per-class accuracy from confusion matrix
    per_class_acc = {}
    row_sums = cm_total.sum(axis=1).astype(float)
    for i, cls in enumerate(classes):
        acc_i = (cm_total[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
        name = class_names[cls] if cls < len(class_names) else str(cls)
        per_class_acc[name] = float(acc_i)

    # optional save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(data_path))[0]
        out_json = os.path.join(save_dir, f"{base}_cls_results.json")

        report = {
            "dataset": meta,
            "settings": {
                "optimizer": optimizer,
                "num_reps": num_reps,
                "kfold": kfold,
                "strict_fold_preprocess": strict_fold_preprocess,
                "use_pca": use_pca,
                "pca_n_components": None if pca_n_components is None else float(_maybe_cast_pca_n(pca_n_components)),
                "pca_whiten": pca_whiten,
                "n_features_final": int(n_feat_final),
                "explained_variance_ratio_cum": None if 'evr_cum' not in locals() else evr_cum,
                "limit_samples": limit_samples,
                "feature_map": feature_map_name or "DEFAULT",
                "ansatz": ansatz_name or "DEFAULT",
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
                "class_names": [class_names[c] if c < len(class_names) else str(c) for c in classes],
            },
        }
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved metrics: {out_json}")

    return acc_mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="One or more .npz dataset paths.")
    ap.add_argument("--optimizer", default="COBYLA")
    ap.add_argument("--num-reps", type=int, default=2)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # PCA
    ap.add_argument("--use-pca", action="store_true")
    ap.add_argument("--pca-n-components", default=None, type=_maybe_cast_pca_n,
                    help="int or fraction in (0,1]; e.g., 4 or 0.95")
    ap.add_argument("--pca-whiten", action="store_true")

    ap.add_argument("--save-dir", default=None, help="Where to write JSON outputs.")
    ap.add_argument("--limit-samples", type=int, default=None, help="Subsample rows for speed.")
    ap.add_argument("--strict-fold-preprocess", action="store_true", help="Fit scaler/PCA inside each fold.")

    # circuit variants (optional)
    ap.add_argument("--feature-map", default=None, help="ZZ | ZZRZZ | ZZPoly")
    ap.add_argument("--ansatz", default=None, help="RA | RAAlt | RACZ")

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
            limit_samples=args.limit_samples,
            strict_fold_preprocess=args.strict_fold_preprocess,
            feature_map_name=args.feature_map,
            ansatz_name=args.ansatz,
        )

if __name__ == "__main__":
    main()
