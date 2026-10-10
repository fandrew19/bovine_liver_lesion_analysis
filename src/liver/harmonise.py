"""Phase 3 -- force every image through one identical capture path.

The two class folders were produced by different export pipelines, leaving the class
label recoverable from JPEG quantization tables, resolution, EXIF and aspect ratio
(finding F-01). Decoding, rotating and cropping every image through the same code --
then storing the result losslessly -- removes those differences.

What this cannot remove: the first compression is already baked into the pixels.
Re-encoding does not erase it, it compounds it (see config.HARMONISE_SIZE). The
residual is measured by probes.compression_artefact_features and reported rather
than hidden.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from .config import HARMONISE_SIZE, HARMONISED_ROOT


def harmonise_image(path: Path, size: int = HARMONISE_SIZE) -> Image.Image:
    """Decode, apply EXIF rotation, centre-crop to square, scale to `size`.

    ImageOps.fit crops to the target aspect ratio before scaling, so a 0.60-ratio
    lesion photo and a 0.75-ratio normal photo are not squashed by different
    amounts (finding F-10). EXIF rotation is applied here and only here, fixing
    finding F-06 -- the normal images carry an orientation tag and the lesion
    images do not, so skipping this step feeds one class in sideways.
    """
    with Image.open(path) as opened:
        img = ImageOps.exif_transpose(opened).convert("RGB")
        return ImageOps.fit(img, (size, size), method=Image.LANCZOS)


def build_harmonised_set(
    manifest: pd.DataFrame,
    data_root: Path,
    out_root: Path = HARMONISED_ROOT,
    force: bool = False,
) -> pd.DataFrame:
    """Write the harmonised copy once, as PNG, so training never re-encodes.

    Returns the manifest with a `harmonised_path` column appended.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    harmonised_paths = []

    for _, row in manifest.iterrows():
        src = data_root / row["path"]
        # liver_id prefix keeps sibling views of one animal adjacent on disk.
        dest = out_root / f"{row['liver_id']}__{Path(row['path']).stem}.png"
        if force or not dest.exists():
            harmonise_image(src).save(dest, "PNG", optimize=True)
        harmonised_paths.append(str(dest.relative_to(out_root.parent)))

    out = manifest.copy()
    out["harmonised_path"] = harmonised_paths
    return out
