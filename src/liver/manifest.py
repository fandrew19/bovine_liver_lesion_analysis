"""Build the dataset manifest: the single source of truth for every later stage.

Replaces folder-walking with one CSV carrying path, labels, liver identity and the
capture-pipeline properties we need to audit (F-01, F-03, F-05).
"""
from __future__ import annotations

import hashlib
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Folder name -> tidy lesion label. Folder casing on disk is inconsistent.
LESION_FROM_FOLDER = {
    "normal_liver": "normal",
    "Liver Fascioliasis": "fascioliasis",
    "Liver Abscesses": "abscess",
    "LIVER Telangiectasis": "telangiectasis",
    "LIVER CIRRHOSIS": "cirrhosis",
    "LIVER FIBROSIS": "fibrosis",
}

# "LF13a" -> "LF13", "ABS 11 a" -> "ABS 11", "T1b" -> "T1".
# A trailing single letter is a view suffix (different face/angle of one liver),
# but only when a digit precedes it -- otherwise "LFF" would collapse into "LF".
_VIEW_SUFFIX = re.compile(r"^(?P<stem>.*\d)\s*[a-z]$", re.IGNORECASE)


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def case_id_from_filename(path: Path) -> str | None:
    """Case identifier encoded in the filename, or None if it carries no number."""
    stem = path.stem.strip()
    m = _VIEW_SUFFIX.match(stem)
    base = m.group("stem") if m else stem
    base = re.sub(r"\s+", " ", base).strip().upper()
    return base if any(ch.isdigit() for ch in base) else None


def _dhash(path: Path, size: int = 16) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    img = img.resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=int)
    return (a[:, 1:] > a[:, :-1]).ravel()


def _thumb(path: Path, size: int = 32) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = ImageOps.fit(img, (size, size), method=Image.LANCZOS)
    a = np.asarray(img, dtype=float).ravel()
    return (a - a.mean()) / (a.std() + 1e-9)


def cluster_by_appearance(
    paths: list[Path], hamming_max: int = 32, corr_min: float = 0.80
) -> dict[Path, int]:
    """Group near-identical photographs -- repeat views of one liver.

    Two images join the same group when their perceptual hashes are close OR their
    normalised thumbnails correlate strongly. Union-find merges the pairs so that
    a chain of similar views ends up as one group.
    """
    hashes = {p: _dhash(p) for p in paths}
    thumbs = {p: _thumb(p) for p in paths}

    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(paths, 2):
        ham = int((hashes[a] != hashes[b]).sum())
        if ham > 45:
            continue
        corr = float(np.dot(thumbs[a], thumbs[b]) / len(thumbs[a]))
        if ham <= hamming_max or corr >= corr_min:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups = defaultdict(list)
    for p in paths:
        groups[find(p)].append(p)
    return {p: i for i, members in enumerate(groups.values()) for p in members}


def _jpeg_quantization_mean(img: Image.Image) -> float:
    tables = getattr(img, "quantization", None)
    if not tables:
        return float("nan")
    return float(np.mean(tables[0]))


def scan(data_root: Path) -> pd.DataFrame:
    """Read every image once, recording labels and capture-pipeline properties."""
    rows = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(data_root)
        top = rel.parts[0]
        folder = top if top == "normal_liver" else rel.parts[1]
        lesion = LESION_FROM_FOLDER.get(folder, folder.lower())

        with Image.open(path) as img:
            width, height = img.size
            exif = img.getexif() or {}
            rows.append(
                {
                    "path": str(rel),
                    "label": 0 if top == "normal_liver" else 1,
                    "lesion_type": lesion,
                    "case_id": case_id_from_filename(path),
                    "md5": md5(path),
                    "width": width,
                    "height": height,
                    "bytes": path.stat().st_size,
                    "exif_orientation": exif.get(274),
                    "qt_mean": _jpeg_quantization_mean(img),
                }
            )
    return pd.DataFrame(rows)


def assign_liver_ids(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Attach a liver_id so cross-validation can keep one animal inside one fold.

    Lesion filenames encode the case directly (LF13a/b/c -> LF13). Normal images are
    numbered sequentially with no case information, so their grouping is recovered by
    clustering appearance. Appearance clustering also merges lesion cases whose
    filenames disagree but whose photographs are near-identical.
    """
    df = df.copy()
    cluster = cluster_by_appearance([data_root / p for p in df["path"]])
    df["appearance_group"] = [cluster[data_root / p] for p in df["path"]]

    # Seed from the filename case where present, otherwise from appearance.
    # Test the type, not truthiness: pandas turns a missing case_id into NaN, and
    # NaN is both truthy and unequal to itself -- which would hang the union-find.
    seed = [
        cid if isinstance(cid, str) and cid else f"APPEAR-{grp}"
        for cid, grp in zip(df["case_id"], df["appearance_group"])
    ]

    # Union filename-cases with appearance-groups: either signal is enough to bind
    # two images to one liver.
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for s in seed:
        find(s)
    for grp, members in pd.Series(seed).groupby(df["appearance_group"].values):
        first = members.iloc[0]
        for other in members.iloc[1:]:
            union(first, other)

    roots = [find(s) for s in seed]
    ordering = {r: i for i, r in enumerate(pd.unique(pd.Series(roots)))}
    df["liver_id"] = [f"L{ordering[r]:03d}" for r in roots]
    return df.drop(columns=["appearance_group"])


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop byte-identical files, keeping the first by path. Returns (kept, dropped)."""
    first = df.sort_values("path").drop_duplicates("md5", keep="first")
    dropped = df[~df.index.isin(first.index)].copy()
    return first.sort_values("path").reset_index(drop=True), dropped


def build(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full manifest pipeline. Returns (manifest, dropped_duplicates)."""
    raw = scan(data_root)
    kept, dropped = deduplicate(raw)
    manifest = assign_liver_ids(kept, data_root)
    columns = [
        "path", "label", "lesion_type", "liver_id", "case_id", "md5",
        "width", "height", "bytes", "exif_orientation", "qt_mean",
    ]
    return manifest[columns], dropped
