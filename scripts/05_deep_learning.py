"""Phase 5 (deep arm) -- three backbones, frozen and fine-tuned (finding F-07).

Frozen configurations reuse precomputed embeddings, so all three run in minutes.
Fine-tuning is genuinely expensive on a 4-thread CPU and is opt-in via --finetune.

    python 05_deep_learning.py                # frozen only (fast)
    python 05_deep_learning.py --finetune     # full 3 x 2 grid (hours on CPU)
"""
import _bootstrap  # noqa: F401

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch

from liver import evaluate, models, splits, train
from liver.config import MANIFEST_CSV, OUTPUTS, PROJECT_ROOT, SEED, ensure_dirs

N_SPLITS = 5
N_REPEATS = 1
N_AUG_VIEWS = 6
SENSITIVITY_TARGET = 0.90


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true",
                        help="also run the fine-tuned configurations (slow on CPU)")
    parser.add_argument("--backbones", nargs="*", default=list(models.BACKBONES))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    torch.set_num_threads(4)
    ensure_dirs()

    manifest = pd.read_csv(MANIFEST_CSV)
    y = manifest["label"].to_numpy()
    groups = manifest["liver_id"].to_numpy()
    paths = [PROJECT_ROOT / p for p in manifest["harmonised_path"]]

    folds = splits.nested_folds(y, groups, N_SPLITS, N_REPEATS, seed=SEED)
    print(f"{len(manifest)} images / {manifest.liver_id.nunique()} livers, "
          f"{len(folds)} nested group folds\n")

    fold_rows, oof_rows = [], []

    def record(model_name, fold, inner_prob, test_prob):
        thr = evaluate.threshold_for_sensitivity(
            y[fold.inner_idx], inner_prob, SENSITIVITY_TARGET
        )
        metrics = evaluate.compute_metrics(y[fold.test_idx], test_prob, thr)
        fold_rows.append({"model": model_name, "split": fold.name, **metrics})
        for i, idx in enumerate(fold.test_idx):
            oof_rows.append({
                "model": model_name, "split": fold.name, "repeat": fold.repeat,
                "row": int(idx), "liver_id": groups[idx], "y_true": int(y[idx]),
                "y_prob": float(test_prob[i]), "threshold": thr,
            })

    for backbone in args.backbones:
        name = f"{backbone} (frozen)"
        t0 = time.time()
        print(f"  embedding {backbone} ({N_AUG_VIEWS} views) ...", flush=True)
        emb = models.compute_embeddings(backbone, paths, n_views=N_AUG_VIEWS, seed=SEED)
        for fold in folds:
            result = train.train_frozen(emb, y, fold)
            record(name, fold, result.inner_prob, result.test_prob)
        aucs = [r["auc"] for r in fold_rows if r["model"] == name]
        print(f"    {name:28s} AUC {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}"
              f"   [{time.time()-t0:.0f}s]")

    if args.finetune:
        for backbone in args.backbones:
            name = f"{backbone} (fine-tuned)"
            t0 = time.time()
            print(f"  fine-tuning {backbone} ...", flush=True)
            for fold in folds:
                result = train.train_finetune(backbone, paths, y, fold)
                record(name, fold, result.inner_prob, result.test_prob)
                print(f"    {fold.name} done ({result.epochs_run} epochs)", flush=True)
            aucs = [r["auc"] for r in fold_rows if r["model"] == name]
            print(f"    {name:28s} AUC {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}"
                  f"   [{time.time()-t0:.0f}s]")

    per_fold = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    per_fold.to_csv(OUTPUTS / "deep_fold_metrics.csv", index=False)
    oof.to_csv(OUTPUTS / "deep_oof_predictions.csv", index=False)

    print("\n--- mean ± SD across folds ---")
    print(evaluate.summarise_folds(per_fold).to_string(index=False))

    print("\n--- pooled out-of-fold, 95% CI bootstrapped over livers ---")
    pooled = evaluate.pooled_report(oof, seed=SEED)
    pooled.to_csv(OUTPUTS / "deep_pooled.csv", index=False)
    for _, r in pooled.iterrows():
        print(f"  {r['model']:28s} AUC {r['auc']:.3f} "
              f"[{r['auc_lo']:.3f}-{r['auc_hi']:.3f}]  sens {r['sensitivity']:.3f}  "
              f"spec {r['specificity']:.3f}")

    print("\nNOTE: outputs/confound_diagnostics.csv shows an 8x8 thumbnail of these")
    print("images scores AUC 0.989. Treat every number above as an upper bound that")
    print("mostly reflects capture conditions, not pathology.")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    run()
