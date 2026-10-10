"""Phase 6 -- is the signal in the lesion, or in how the photograph was taken?

The classical baseline reaches AUC ~0.99 on harmonised images. That is either real
pathology or residual framing/lighting confound. These diagnostics separate the two
by deliberately destroying lesion detail and seeing what skill survives.

A probe that stays high after the lesion is unreadable is measuring something else.
"""
import _bootstrap  # noqa: F401

import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from liver.config import MANIFEST_CSV, OUTPUTS, PROJECT_ROOT, SEED, ensure_dirs


def colour_stats(arr: np.ndarray) -> list[float]:
    out = []
    for c in range(arr.shape[2]):
        ch = arr[:, :, c].astype(float)
        out.extend([ch.mean(), ch.std(), np.percentile(ch, 10), np.percentile(ch, 90)])
    return out


def border_ring(a: np.ndarray, frac: float = 0.20) -> np.ndarray:
    """The outer frame, with the centre discarded. Mostly table, hands, background."""
    n = a.shape[0]
    k = int(n * frac)
    return np.concatenate([
        a[:k].reshape(-1, 3), a[-k:].reshape(-1, 3),
        a[k:-k, :k].reshape(-1, 3), a[k:-k, -k:].reshape(-1, 3),
    ])[None, :, :]


def variant(img: Image.Image, kind: str) -> np.ndarray:
    """Each variant removes a different kind of information."""
    a = np.asarray(img.convert("RGB"))
    n = a.shape[0]

    if kind == "full image":
        return np.asarray(colour_stats(a))

    if kind == "border ring only (no centre)":
        return np.asarray(colour_stats(border_ring(a)))

    if kind == "border ring, per-image colour normalised":
        # Remove each photograph's own colour cast and exposure, then look at the
        # border again. If the border still predicts the class, the difference between
        # the classes is framing and composition, not merely a global colour shift --
        # so per-image colour normalisation cannot be the remedy.
        flat = a.reshape(-1, 3).astype(float)
        z = (a.astype(float) - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-6)
        return np.asarray(colour_stats(border_ring(z)))

    if kind == "centre patch only":
        k = int(n * 0.30)
        return np.asarray(colour_stats(a[k:-k, k:-k]))

    if kind == "heavy blur (colour layout only)":
        blurred = img.convert("RGB").filter(ImageFilter.GaussianBlur(12))
        return np.asarray(colour_stats(np.asarray(blurred)))

    if kind == "8x8 thumbnail (all detail gone)":
        tiny = img.convert("RGB").resize((8, 8), Image.BOX)
        return np.asarray(tiny, dtype=float).ravel()

    if kind == "mean colour only (3 numbers)":
        return a.reshape(-1, 3).mean(axis=0)

    raise ValueError(kind)


VARIANTS = [
    "full image",
    "centre patch only",
    "border ring only (no centre)",
    "border ring, per-image colour normalised",
    "heavy blur (colour layout only)",
    "8x8 thumbnail (all detail gone)",
    "mean colour only (3 numbers)",
]


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    manifest = pd.read_csv(MANIFEST_CSV)
    y = manifest["label"].to_numpy()
    groups = manifest["liver_id"].to_numpy()
    images = [Image.open(PROJECT_ROOT / p).copy()
              for p in manifest["harmonised_path"]]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    )

    print("How much class signal survives when lesion detail is destroyed?\n")
    print(f"  {'variant':36s} {'AUC':>7s}   interpretation")
    print("  " + "-" * 74)

    rows = []
    for kind in VARIANTS:
        X = np.vstack([variant(im, kind) for im in images])
        auc = cross_val_score(
            model, X, y, groups=groups, cv=cv, scoring="roc_auc"
        ).mean()
        if kind == "full image":
            note = "reference"
        elif auc >= 0.85:
            note = "high -- signal is NOT lesion detail"
        elif auc >= 0.70:
            note = "moderate residual"
        else:
            note = "low -- detail mattered"
        rows.append({"variant": kind, "auc": auc, "note": note})
        print(f"  {kind:36s} {auc:7.3f}   {note}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "confound_diagnostics.csv", index=False)

    ring = df.loc[df.variant == "border ring only (no centre)", "auc"].item()
    tiny = df.loc[df.variant.str.startswith("8x8"), "auc"].item()
    mean_only = df.loc[df.variant.str.startswith("mean colour"), "auc"].item()

    print("\nReading:")
    if ring >= 0.75:
        print(f"  * Border ring scores {ring:.3f}. The area around the organ predicts the")
        print("    class, so background, lighting or surface differ systematically between")
        print("    the two classes. This is a capture confound, not pathology.")
    else:
        print(f"  * Border ring only {ring:.3f} -- background is not carrying the label.")
    if mean_only >= 0.80:
        print(f"  * Three average colour numbers score {mean_only:.3f}. Overall colour cast")
        print("    alone nearly solves the task; the CNN need not learn lesion morphology.")
    if tiny >= 0.85:
        print(f"  * An 8x8 thumbnail scores {tiny:.3f}. At that size no lesion is resolvable,")
        print("    so coarse colour layout is doing the work.")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
