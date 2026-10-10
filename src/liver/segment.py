"""Segment the liver out of the photograph, so analysis can ignore the scene.

The dataset's confound lives in the background and the framing: the image border alone
predicts the class at AUC 0.971. Restricting every measurement to pixels inside the organ
removes the background entirely, and sampling fixed-size patches from the interior removes
the organ's outline and the shot's composition too.

Segmentation is classical rather than learned, deliberately: there are no liver masks to
train on, and a colour/lightness rule is inspectable and cheap. Liver parenchyma is dark
and red-shifted; inspection tables, cream matting and gravel are bright and desaturated.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
from skimage.morphology import (
    binary_closing,
    binary_erosion,
    binary_opening,
    disk,
    remove_small_objects,
)


def liver_mask(
    img: Image.Image, n_colours: int = 4, min_fraction: float = 0.08, seed: int = 0
) -> np.ndarray:
    """Boolean mask of the organ.

    A plain darkness or redness threshold keeps too much: pooled blood and wet table are
    both dark and red. Clustering Lab colour instead lets parenchyma, blood, matting and
    shadow separate on their own, and the organ is then picked out as the cluster with the
    most mass near the frame centre -- the subject of the photograph is what the
    photographer aimed at.
    """
    arr = np.asarray(img.convert("RGB"))
    lab = rgb2lab(arr)
    h, w = lab.shape[:2]

    # Cluster on a downsampled copy; full resolution adds nothing and costs time.
    small = lab[::3, ::3].reshape(-1, 3)
    km = KMeans(n_clusters=n_colours, n_init=4, random_state=seed).fit(small)
    labels = km.predict(lab.reshape(-1, 3)).reshape(h, w)

    # Weight each pixel by closeness to the centre, then score clusters by total weight.
    yy, xx = np.mgrid[0:h, 0:w]
    centre_weight = np.exp(
        -(((yy - h / 2) / (h * 0.45)) ** 2 + ((xx - w / 2) / (w * 0.45)) ** 2)
    )
    scores = [
        centre_weight[labels == k].sum() if (labels == k).any() else -1.0
        for k in range(n_colours)
    ]
    mask = labels == int(np.argmax(scores))

    # Merge in any cluster that is darker than the chosen one and touches it: livers span
    # several shades once cut, and a fissure should not split the organ in two.
    chosen_l = lab[:, :, 0][mask].mean() if mask.any() else 50.0
    for k in range(n_colours):
        if labels_k_mergeable(labels, k, mask, lab, chosen_l):
            mask |= labels == k

    mask = binary_opening(mask, disk(3))
    mask = binary_closing(mask, disk(9))
    mask = ndi.binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=int(mask.size * 0.02))

    components, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, components, range(1, n + 1))
        mask = components == (int(np.argmax(sizes)) + 1)

    # A mask covering almost nothing means the rule failed on this image; fall back to a
    # generous centre region rather than returning something empty.
    if mask.mean() < min_fraction:
        mask = np.zeros((h, w), dtype=bool)
        mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = True
    return mask


def labels_k_mergeable(labels, k, mask, lab, chosen_l, max_lightness_gap=18.0) -> bool:
    """True when cluster k is a darker shade of the organ already selected."""
    candidate = labels == k
    if candidate is mask or not candidate.any() or (candidate & mask).any():
        return False
    if not (binary_closing(mask, disk(5)) & candidate).any():
        return False          # not adjacent to the organ
    return 0 <= chosen_l - lab[:, :, 0][candidate].mean() <= max_lightness_gap


def eroded(mask: np.ndarray, margin: int = 12) -> np.ndarray:
    """Pull the mask inward so samples cannot sit on the organ's outline."""
    inner = binary_erosion(mask, disk(margin))
    return inner if inner.mean() > 0.02 else mask


def normalise_tissue_colour(
    arr: np.ndarray, mask: np.ndarray, target_l: float = 45.0, target_scale: float = 12.0
) -> np.ndarray:
    """Standardise lightness *within the organ* to a common reference.

    Sessions differed in lighting and white balance, so raw tissue colour partly encodes
    when the photograph was taken. Normalising against the organ's own statistics removes
    that per-image offset while leaving local texture intact.
    """
    lab = rgb2lab(arr)
    inside = lab[:, :, 0][mask]
    if inside.size < 10:
        return lab[:, :, 0]
    standardised = (lab[:, :, 0] - inside.mean()) / (inside.std() + 1e-6)
    return standardised * target_scale + target_l


def sample_patches(
    img: Image.Image,
    mask: np.ndarray,
    patch: int = 64,
    n_patches: int = 12,
    coverage: float = 0.92,
    seed: int = 0,
) -> list[np.ndarray]:
    """Fixed-size windows drawn from well inside the organ.

    A patch is accepted only when `coverage` of it falls within the mask, so no sample
    straddles the organ outline -- the boundary is part of the composition we are trying
    to discard.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB"))
    h, w = mask.shape
    if h <= patch or w <= patch:
        return []

    integral = mask.cumsum(0).cumsum(1)

    def covered(top, left):
        bottom, right = top + patch, left + patch
        total = integral[bottom - 1, right - 1]
        if top:
            total -= integral[top - 1, right - 1]
        if left:
            total -= integral[bottom - 1, left - 1]
        if top and left:
            total += integral[top - 1, left - 1]
        return total / (patch * patch)

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []

    patches, attempts = [], 0
    while len(patches) < n_patches and attempts < n_patches * 60:
        attempts += 1
        i = rng.integers(len(ys))
        top = int(np.clip(ys[i] - patch // 2, 0, h - patch))
        left = int(np.clip(xs[i] - patch // 2, 0, w - patch))
        if covered(top, left) >= coverage:
            patches.append(arr[top:top + patch, left:left + patch])
    return patches


def organ_relative_patches(
    img: Image.Image,
    mask: np.ndarray,
    fraction: float = 0.30,
    out_size: int = 64,
    n_patches: int = 12,
    seed: int = 0,
) -> list[np.ndarray]:
    """Sample windows covering a fixed fraction of the ORGAN, then resize to `out_size`.

    A close-up and a whole-organ shot show tissue at different magnifications, so a
    fixed-pixel window means different things in each. Sizing the window against the
    organ's own apparent area normalises that, and rescaling to a common output size
    puts every patch at comparable tissue scale.
    """
    side = int(np.clip(fraction * np.sqrt(mask.sum()), 24, min(mask.shape) - 1))
    for margin, coverage in ((max(4, side // 8), 0.85), (0, 0.60)):
        found = sample_patches(
            img, eroded(mask, margin) if margin else mask,
            patch=side, n_patches=n_patches, coverage=coverage, seed=seed,
        )
        if len(found) >= max(3, n_patches // 3):
            break
    return [
        np.asarray(Image.fromarray(p).resize((out_size, out_size), Image.LANCZOS))
        for p in found
    ]


def patches_for_image(
    img: Image.Image, patch: int = 64, n_patches: int = 12, seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Sample interior patches, relaxing the constraints if the organ is small.

    A tightly cropped photograph can leave an eroded mask too thin to hold a 64-pixel
    window. Rather than drop the image, step the erosion margin and coverage requirement
    down until patches can be placed.
    """
    mask = liver_mask(img, seed=seed)
    for margin, coverage in ((12, 0.92), (6, 0.85), (0, 0.75), (0, 0.55)):
        found = sample_patches(
            img, eroded(mask, margin) if margin else mask,
            patch=patch, n_patches=n_patches, coverage=coverage, seed=seed,
        )
        if len(found) >= max(3, n_patches // 3):
            return found, mask
    return found, mask
