"""Phase 5 (classical arm) -- SVM and Random Forest on colour + texture features.

Uses the same nested group folds the deep models will use, so the comparison
required by proposal section 2.6 is like-for-like.
"""
import _bootstrap  # noqa: F401

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from liver import evaluate, features, splits
from liver.config import MANIFEST_CSV, OUTPUTS, PROJECT_ROOT, SEED, ensure_dirs

N_SPLITS = 5
N_REPEATS = 3
SENSITIVITY_TARGET = 0.90


def build_models() -> dict:
    return {
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced",
                probability=False, random_state=SEED),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        ),
        "Logistic (L2)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1),
        ),
    }


def scores(model, X) -> np.ndarray:
    """Continuous score for ROC. decision_function where available, else P(class 1)."""
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict_proba(X)[:, 1]


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    manifest = pd.read_csv(MANIFEST_CSV)
    y = manifest["label"].to_numpy()
    groups = manifest["liver_id"].to_numpy()
    paths = [PROJECT_ROOT / p for p in manifest["harmonised_path"]]

    print(f"Extracting colour + texture features for {len(paths)} images ...")
    X = features.extract_all(paths)
    print(f"  feature matrix: {X.shape}\n")

    folds = splits.nested_folds(y, groups, N_SPLITS, N_REPEATS, seed=SEED)
    print(splits.describe(folds[:N_SPLITS], y, groups))
    print(f"\n({len(folds)} splits total = {N_REPEATS} repeats x {N_SPLITS} folds)\n")

    fold_rows, oof_rows = [], []
    for name, template in build_models().items():
        for f in folds:
            model = __import__("sklearn").base.clone(template)
            model.fit(X[f.train_idx], y[f.train_idx])

            # Operating point chosen on the inner fold only.
            thr = evaluate.threshold_for_sensitivity(
                y[f.inner_idx], scores(model, X[f.inner_idx]), SENSITIVITY_TARGET
            )
            p_test = scores(model, X[f.test_idx])
            metrics = evaluate.compute_metrics(y[f.test_idx], p_test, thr)
            fold_rows.append({"model": name, "split": f.name, **metrics})

            for i, idx in enumerate(f.test_idx):
                oof_rows.append({
                    "model": name, "split": f.name, "repeat": f.repeat,
                    "row": int(idx), "liver_id": groups[idx],
                    "y_true": int(y[idx]), "y_prob": float(p_test[i]),
                    "threshold": thr,
                })
        done = pd.DataFrame(fold_rows).query("model == @name")
        print(f"  {name:16s} fold AUC {np.nanmean(done.auc):.3f} ± {np.nanstd(done.auc):.3f}")

    per_fold = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    per_fold.to_csv(OUTPUTS / "classical_fold_metrics.csv", index=False)
    oof.to_csv(OUTPUTS / "classical_oof_predictions.csv", index=False)

    print("\n--- mean ± SD across folds ---")
    print(evaluate.summarise_folds(per_fold).to_string(index=False))

    # Pool one repeat at a time, then report the first; repeats differ only by split seed.
    print("\n--- pooled out-of-fold, 95% CI bootstrapped over livers ---")
    pooled = evaluate.pooled_report(oof[oof.repeat == 0], seed=SEED)
    pooled.to_csv(OUTPUTS / "classical_pooled.csv", index=False)
    for _, r in pooled.iterrows():
        print(f"  {r['model']:16s} AUC {r['auc']:.3f} [{r['auc_lo']:.3f}-{r['auc_hi']:.3f}]  "
              f"sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
