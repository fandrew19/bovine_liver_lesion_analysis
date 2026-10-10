"""Does magnification normalisation help, or is it doing the work?

Script 07 sizes each patch window against the organ's own apparent area, so a close-up
and a whole-organ shot contribute tissue at comparable magnification. This script runs
the same texture analysis with that normalisation switched OFF -- a fixed 64-pixel
window regardless of organ size -- and reports both.

The comparison matters for interpretation. A signal that is genuinely tissue structure
should *improve* when magnification is equalised, because the equalisation removes a
nuisance source of variation. A signal that is really an artefact of camera distance
should degrade, because that is precisely the information being taken away.
"""
import _bootstrap  # noqa: F401

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from liver import segment, splits
from liver.config import MANIFEST_CSV, OUTPUTS, PROJECT_ROOT, SEED, ensure_dirs

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _load_07():
    """Reuse script 07's feature and scoring code (its name is not a valid identifier)."""
    path = Path(__file__).resolve().parent / "07_patch_analysis.py"
    spec = importlib.util.spec_from_file_location("patch_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    pa = _load_07()

    manifest = pd.read_csv(MANIFEST_CSV)
    labels = manifest["label"].to_numpy()
    groups = manifest["liver_id"].to_numpy()
    folds = splits.nested_folds(labels, groups, n_splits=5, n_repeats=1, seed=SEED)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    )

    # Two sampling regimes over identical images, masks, seed and classifier. The only
    # difference is whether the window is sized against the organ or fixed in pixels.
    regimes = {
        "organ-relative (magnification normalised)":
            lambda img, mask: segment.organ_relative_patches(
                img, mask, out_size=pa.PATCH, n_patches=pa.N_PATCHES, seed=SEED),
        "fixed 64 px window (not normalised)":
            lambda img, mask: pa.segment.patches_for_image(
                img, patch=pa.PATCH, n_patches=pa.N_PATCHES, seed=SEED)[0],
    }

    rows = []
    for name, sampler in regimes.items():
        patches, owner = [], []
        for row, rel in enumerate(manifest["harmonised_path"]):
            img = Image.open(PROJECT_ROOT / rel)
            mask = segment.liver_mask(img, seed=SEED)
            found = sampler(img, mask)
            patches.extend(found)
            owner.extend([row] * len(found))

        owner = np.asarray(owner)
        X = np.vstack([pa.texture_features(p) for p in patches])
        auc, lo, hi = pa.liver_level_auc(model, X, labels[owner], owner, folds, groups)
        rows.append({"sampling": name, "auc": auc, "lo": lo, "hi": hi,
                     "n_patches": len(patches)})
        print(f"  {name:45s} {auc:.3f}  [{lo:.3f}-{hi:.3f}]  ({len(patches)} patches)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "scale_ablation.csv", index=False)

    norm = df.loc[df.sampling.str.startswith("organ"), "auc"].item()
    fixed = df.loc[df.sampling.str.startswith("fixed"), "auc"].item()
    print(f"\n  Normalising magnification changes texture AUC by {norm - fixed:+.3f}.")
    print("  An improvement argues the signal is tissue structure, not camera distance."
          if norm > fixed else
          "  A decrease would suggest magnification was carrying part of the signal.")


if __name__ == "__main__":
    main()
