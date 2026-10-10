"""Objective 5 -- does the model look at the lesion, or at the table?

Trains the frozen-backbone head on one fold, folds the scaler and logistic
coefficients into a torch Linear layer so the whole thing is one differentiable
network, then measures where the evidence lives:

  * Grad-CAM border share -- fraction of activation in the outer 20% frame.
  * Occlusion             -- how much the prediction moves when a region is hidden.

A uniform attention map puts 0.64 of its mass in that frame, so ~0.64 means
"no preference". Substantially above means the model is reading the background.
"""
import _bootstrap  # noqa: F401

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from liver import gradcam, models, splits
from liver.config import MANIFEST_CSV, OUTPUTS, PROJECT_ROOT, SEED, ensure_dirs

BACKBONE = "mobilenet_v2"
N_EXAMPLES = 6


def fold_head(embeddings, y, fold) -> nn.Linear:
    """Fit the frozen head, then fold StandardScaler into the Linear layer.

        z = w . ((x - mu) / sigma) + b  =  (w/sigma) . x + (b - sum(w*mu/sigma))
    """
    X = np.vstack([v[fold.train_idx] for v in embeddings])
    y_train = np.tile(y[fold.train_idx], embeddings.shape[0])
    scaler = StandardScaler().fit(X)
    head = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced",
                              random_state=SEED).fit(scaler.transform(X), y_train)

    w = head.coef_[0] / scaler.scale_
    b = head.intercept_[0] - float(np.dot(head.coef_[0], scaler.mean_ / scaler.scale_))

    layer = nn.Linear(len(w), 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(w, dtype=torch.float32).unsqueeze(0))
        layer.bias.copy_(torch.tensor([b], dtype=torch.float32))
    return layer


def assemble(backbone: str, head: nn.Linear) -> nn.Module:
    model = models.build(backbone, pretrained=True, freeze=True)
    model.classifier = head if backbone != "resnet18" else model.classifier
    if backbone == "resnet18":
        model.fc = head
    return model.eval()


def main() -> None:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(4)
    ensure_dirs()

    manifest = pd.read_csv(MANIFEST_CSV)
    y = manifest["label"].to_numpy()
    groups = manifest["liver_id"].to_numpy()
    paths = [PROJECT_ROOT / p for p in manifest["harmonised_path"]]

    folds = splits.nested_folds(y, groups, 5, 1, seed=SEED)
    fold = folds[0]

    print(f"Embedding {BACKBONE} ...", flush=True)
    emb = models.compute_embeddings(BACKBONE, paths, n_views=6, seed=SEED)
    model = assemble(BACKBONE, fold_head(emb, y, fold))
    layer = models.target_layer(model, BACKBONE)

    print(f"Measuring attention on {len(fold.test_idx)} held-out images ...\n")
    rows = []
    for idx in fold.test_idx:
        x, _ = gradcam.load_image(paths[idx])
        cam, prob = gradcam.grad_cam(model, layer, x)
        rows.append({
            "path": manifest.loc[idx, "path"],
            "lesion_type": manifest.loc[idx, "lesion_type"],
            "y_true": int(y[idx]),
            "prob": prob,
            "border_share": gradcam.attention_in_border(cam),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "gradcam_attention.csv", index=False)

    uniform = 1 - 0.6 ** 2
    print(f"  Grad-CAM attention in the outer 20% frame")
    print(f"    uniform-map reference : {uniform:.3f}")
    print(f"    normal images         : {df[df.y_true == 0].border_share.mean():.3f}")
    print(f"    lesion images         : {df[df.y_true == 1].border_share.mean():.3f}")
    print(f"    all                   : {df.border_share.mean():.3f}")

    # Occlusion on a handful of confident calls from each class.
    print(f"\n  Occlusion test on {N_EXAMPLES} images ...", flush=True)
    picks = pd.concat([
        df[df.y_true == 1].nlargest(N_EXAMPLES // 2, "prob"),
        df[df.y_true == 0].nsmallest(N_EXAMPLES // 2, "prob"),
    ])
    occ_rows = []
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    for ax_row, (_, r) in zip(np.atleast_2d(axes), picks.iterrows()):
        full = PROJECT_ROOT / manifest.loc[
            manifest.path == r["path"], "harmonised_path"].iloc[0]
        x, img = gradcam.load_image(full)
        cam, _ = gradcam.grad_cam(model, layer, x)
        heat, base = gradcam.occlusion_map(model, x)
        occ_rows.append({
            "path": r["path"], "y_true": r["y_true"], "base_prob": base,
            "occlusion_border_share": gradcam.attention_in_border(
                np.clip(heat, 0, None) / (np.abs(heat).max() + 1e-9)
            ),
        })
        ax_row[0].imshow(img); ax_row[0].set_title(
            f"{r['lesion_type']}\nP={base:.2f}", fontsize=9)
        ax_row[1].imshow(gradcam.overlay(img, cam)); ax_row[1].set_title(
            f"Grad-CAM (border {r['border_share']:.2f})", fontsize=9)
        lim = np.abs(heat).max() + 1e-9
        ax_row[2].imshow(heat, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax_row[2].set_title("Occlusion (red = evidence)", fontsize=9)
        for ax in ax_row:
            ax.axis("off")
    plt.tight_layout()
    out_png = OUTPUTS / "figures" / "gradcam_occlusion.png"
    plt.savefig(out_png, dpi=140)
    plt.close()

    occ = pd.DataFrame(occ_rows)
    occ.to_csv(OUTPUTS / "occlusion_attention.csv", index=False)
    print(f"    occlusion evidence in outer frame: "
          f"{occ.occlusion_border_share.mean():.3f} (uniform {uniform:.3f})")
    print(f"    figure -> {out_png.relative_to(PROJECT_ROOT)}")

    print("\nReading:")
    if df.border_share.mean() > uniform + 0.05:
        print("  Attention sits OUTSIDE the organ more than chance. Objective 5 FAILS")
        print("  on this dataset: predictions are not lesion-driven.")
    elif df.border_share.mean() < uniform - 0.10:
        print("  Attention concentrates centrally, consistent with lesion-driven")
        print("  prediction. Confirm against inspector review before claiming it.")
    else:
        print("  Attention is close to uniform -- the map is diffuse, which is itself")
        print("  evidence the model has no localised lesion evidence to point at.")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
