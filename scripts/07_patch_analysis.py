"""Salvage attempt -- can anything valid be learned from the images already collected?

The confound lives in the background and the framing. This route discards both: segment
the organ, sample fixed-size patches from well inside it, and classify tissue rather than
photographs. Organ outline, shot composition, inspection surface and image border are all
gone by construction.

Two feature sets are compared, because they fail differently:

  colour   -- patch colour statistics. Sensitive to real pathology (pale fascioliasis
              tracts, white abscess foci) but also to session lighting and white balance.
  texture  -- GLCM on lightness that has been standardised *within the organ*, which
              removes each image's own colour and exposure offset and leaves structure.

Every configuration is scored against its own leakage probes. Patch predictions are
averaged per animal, and cross-validation groups by animal.
"""
import _bootstrap  # noqa: F401

import warnings

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from skimage.feature import graycomatrix, graycoprops
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from liver import evaluate, segment, splits
from liver.config import MANIFEST_CSV, OUTPUTS, PROJECT_ROOT, SEED, ensure_dirs

PATCH = 64
N_PATCHES = 12
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def colour_features(patch: np.ndarray) -> np.ndarray:
    lab = rgb2lab(patch)
    out = []
    for c in range(3):
        ch = lab[:, :, c]
        out.extend([ch.mean(), ch.std(), np.percentile(ch, 10), np.percentile(ch, 90)])
    for c in range(3):
        hist, _ = np.histogram(patch[:, :, c], bins=8, range=(0, 255), density=True)
        out.extend(hist)
    return np.asarray(out, dtype=np.float32)


def texture_features(patch: np.ndarray) -> np.ndarray:
    """GLCM on lightness standardised within the patch.

    Standardising per patch means absolute brightness -- which encodes the session, not
    the pathology -- cannot contribute. Only the spatial arrangement survives.
    """
    lightness = rgb2lab(patch)[:, :, 0]
    z = (lightness - lightness.mean()) / (lightness.std() + 1e-6)
    quantised = np.clip((z * 8 + 32), 0, 63).astype(np.uint8)
    glcm = graycomatrix(
        quantised, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64, symmetric=True, normed=True,
    )
    out = []
    for prop in GLCM_PROPS:
        out.extend(graycoprops(glcm, prop).ravel())
    return np.asarray(out, dtype=np.float32)


def mean_colour_only(patch: np.ndarray) -> np.ndarray:
    """Leakage probe: three numbers, no structure at all."""
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


def scale_only(patch: np.ndarray) -> np.ndarray:
    """Leakage probe: apparent magnification, via the patch's spatial-frequency profile.

    A close-up renders tissue features across more pixels than a whole-organ shot, so
    zoom alone could masquerade as texture. This measures where the patch's energy sits
    in frequency, plus its autocorrelation width -- scale, carrying no pathology.
    """
    a = rgb2lab(patch)[:, :, 0]
    a = (a - a.mean()) / (a.std() + 1e-6)
    power = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    n = power.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    radius = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    profile = np.array([power[(radius >= k) & (radius < k + 1)].mean()
                        for k in range(1, n // 2)])
    profile = profile / (profile.sum() + 1e-9)
    autocorr = np.fft.ifft2(np.abs(np.fft.fft2(a)) ** 2).real
    autocorr = autocorr / (autocorr[0, 0] + 1e-9)
    return np.asarray([
        float((profile * np.arange(1, n // 2)).sum()),
        float((autocorr[0, :n // 2] > 0.5).sum()),
        float(profile[:5].sum()),
        float(profile[-10:].sum()),
    ], dtype=np.float32)


def blockiness_only(patch: np.ndarray) -> np.ndarray:
    """Leakage probe: JPEG block structure, no pathology."""
    a = rgb2lab(patch)[:, :, 0]
    dh = np.abs(np.diff(a, axis=1))
    dv = np.abs(np.diff(a, axis=0))
    return np.asarray([
        dh[:, 7::8].mean() / (np.delete(dh, np.s_[7::8], axis=1).mean() + 1e-9),
        dv[7::8, :].mean() / (np.delete(dv, np.s_[7::8], axis=0).mean() + 1e-9),
    ], dtype=np.float32)


def liver_level_auc(model, X, y, patch_owner, folds, groups_by_row) -> tuple[float, float, float]:
    """Train on patches, predict per patch, average per animal, score per animal."""
    oof = np.full(len(np.unique(patch_owner)), np.nan)
    owners = np.unique(patch_owner)
    owner_pos = {o: i for i, o in enumerate(owners)}
    y_by_owner = np.array([y[patch_owner == o][0] for o in owners])

    for fold in folds:
        train_mask = np.isin(patch_owner, fold.train_idx) | np.isin(patch_owner, fold.inner_idx)
        test_mask = np.isin(patch_owner, fold.test_idx)
        if not train_mask.any() or not test_mask.any():
            continue
        fitted = clone(model).fit(X[train_mask], y[train_mask])
        raw = (fitted.decision_function(X[test_mask])
               if hasattr(fitted, "decision_function")
               else fitted.predict_proba(X[test_mask])[:, 1])
        prob = 1.0 / (1.0 + np.exp(-raw)) if raw.ndim == 1 else raw
        for owner in np.unique(patch_owner[test_mask]):
            sel = patch_owner[test_mask] == owner
            oof[owner_pos[owner]] = float(prob[sel].mean())

    ok = ~np.isnan(oof)
    auc = roc_auc_score(y_by_owner[ok], oof[ok])
    _, lo, hi = evaluate.bootstrap_auc_ci(
        y_by_owner[ok], oof[ok], groups_by_row[owners[ok]], seed=SEED
    )
    return float(auc), lo, hi


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()

    manifest = pd.read_csv(MANIFEST_CSV)
    labels = manifest["label"].to_numpy()
    groups = manifest["liver_id"].to_numpy()

    print(f"Segmenting and sampling patches from {len(manifest)} images "
          f"({PATCH}px, up to {N_PATCHES} each) ...", flush=True)

    patches, owner, coverage = [], [], []
    for row, rel in enumerate(manifest["harmonised_path"]):
        img = Image.open(PROJECT_ROOT / rel)
        mask = segment.liver_mask(img, seed=SEED)
        found = segment.organ_relative_patches(
            img, mask, out_size=PATCH, n_patches=N_PATCHES, seed=SEED
        )
        coverage.append(mask.mean())
        patches.extend(found)
        owner.extend([row] * len(found))

    owner = np.asarray(owner)
    y_patch = labels[owner]
    manifest["mask_fraction"] = coverage
    manifest.to_csv(MANIFEST_CSV, index=False)

    print(f"  {len(patches)} patches from {len(np.unique(owner))} images")
    print(f"  images yielding no patch: {len(manifest) - len(np.unique(owner))}")
    print(f"  mask coverage: normal {np.mean([c for c, l in zip(coverage, labels) if l == 0]):.2f}"
          f"  lesion {np.mean([c for c, l in zip(coverage, labels) if l == 1]):.2f}\n")

    # Persisted so the write-up can quote these without re-running segmentation.
    pd.DataFrame([{
        "n_patches": len(patches),
        "n_images_sampled": int(len(np.unique(owner))),
        "n_images_no_patch": int(len(manifest) - len(np.unique(owner))),
        "patch_px": PATCH,
        "max_patches_per_image": N_PATCHES,
        "mask_frac_normal": float(np.mean([c for c, l in zip(coverage, labels) if l == 0])),
        "mask_frac_lesion": float(np.mean([c for c, l in zip(coverage, labels) if l == 1])),
        "patches_per_image": len(patches) / max(len(np.unique(owner)), 1),
    }]).to_csv(OUTPUTS / "patch_summary.csv", index=False)

    folds = splits.nested_folds(labels, groups, n_splits=5, n_repeats=1, seed=SEED)
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    )
    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.05),
    )

    configs = [
        ("PROBE  patch mean colour only", mean_colour_only, linear, True),
        ("PROBE  patch blockiness only", blockiness_only, linear, True),
        ("PROBE  patch scale / magnification only", scale_only, linear, True),
        ("TASK   patch colour features", colour_features, model, False),
        ("TASK   patch texture only (illumination-invariant)", texture_features, model, False),
    ]

    rows = []
    print(f"  {'configuration':52s} {'liver AUC':>10s}  {'95% CI':>15s}")
    print("  " + "-" * 82)
    for name, extract, estimator, is_probe in configs:
        X = np.vstack([extract(p) for p in patches])
        auc, lo, hi = liver_level_auc(estimator, X, y_patch, owner, folds, groups)
        rows.append({"configuration": name, "auc": auc, "lo": lo, "hi": hi,
                     "probe": is_probe, "n_features": X.shape[1]})
        print(f"  {name:52s} {auc:10.3f}  [{lo:.3f}-{hi:.3f}]")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "patch_analysis.csv", index=False)

    probe_max = df[df.probe].auc.max()
    texture = df[df.configuration.str.contains("texture")].auc.item()
    colour = df[df.configuration.str.contains("colour features")].auc.item()

    print("\nReading:")
    print(f"  Worst leakage probe : {probe_max:.3f}")
    print(f"  Colour features     : {colour:.3f}")
    print(f"  Texture only        : {texture:.3f}")
    print()
    if texture > probe_max + 0.10:
        print("  Texture inside the organ beats every leakage probe by a clear margin.")
        print("  This is a defensible analysis on the existing images: it cannot be")
        print("  explained by background, framing, or session lighting.")
    elif texture > probe_max + 0.05:
        print("  Texture edges ahead of the probes. Suggestive, worth reporting with the")
        print("  probe values alongside it, but not strong enough to stand alone.")
    else:
        print("  Texture does not separate from the leakage probes. The existing images")
        print("  cannot support a classification claim even at patch level.")


if __name__ == "__main__":
    pd.set_option("display.width", 150)
    main()
