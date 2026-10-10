# Bovine Liver Lesion Classification from Smartphone Images

A pilot study on whether a classifier can detect liver lesions in cattle from smartphone
photographs taken during routine post-mortem meat inspection at an abattoir in Kumasi,
Ghana.

**Key finding.** The obvious approach, classifying whole photographs, scores AUC ≈ 0.99
with every model tried. That score is not real: normal and lesioned livers were
photographed under different conditions, so the class can be read from the background
and framing without looking at the liver. This repository documents how the confound was
found, and gives a tissue-only patch analysis that holds up once it is controlled for:

> **Texture-only classification of liver tissue patches: AUC 0.940 (95% CI 0.896–0.973),
> scored per animal across 132 animals.**

## Contents

- [Results](#results)
- [Dataset](#dataset)
- [Installation](#installation)
- [Running the pipeline](#running-the-pipeline)
- [Repository layout](#repository-layout)
- [Acceptance gates for new data](#acceptance-gates-for-new-data)
- [Collection protocol](#collection-protocol)
- [Limitations](#limitations)

## Results

### Whole-image classification is confounded

MobileNetV2, EfficientNet-B0, ResNet18, SVM, Random Forest and logistic regression all
reach AUC ≈ 0.99 on whole images. So does a classifier given almost nothing about the
liver:

| What the classifier was given | AUC |
|---|---|
| File metadata only, no pixels (before harmonisation) | 1.000 |
| 8×8 thumbnail (too small to show a lesion) | 0.989 |
| Border ring only, with the organ removed | 0.971 |
| Border ring, colour-normalised per image | 0.944 |
| Centre patch, where the lesion is | 0.819 |

The border alone predicts the class better than the lesion region does. Grad-CAM agrees:
on normal livers, 99.8% of the model's attention falls outside the organ, against 0.64
for a uniform map. Harmonising image size and encoding, normalising colour per image and
cropping to the centre were each tried, and none removed the confound.

### Tissue-patch analysis holds up

Script 07 removes the background and framing altogether. It segments the organ, samples
patches from well inside it, and classifies tissue instead of photographs. Patch size is
set as a fraction of the organ's apparent area, so close-ups and whole-organ shots give
tissue at comparable magnification. Texture features are computed on lightness
standardised within each patch, so session lighting and white balance cancel out.

Each confound that could survive this step gets its own probe. Scores are per animal,
using grouped cross-validation:

| Features | AUC | 95% CI |
|---|---|---|
| Probe: patch mean colour | 0.715 | 0.625–0.803 |
| Probe: patch JPEG blockiness | 0.689 | 0.600–0.777 |
| Probe: patch scale / magnification | 0.631 | 0.530–0.729 |
| Colour features | 0.909 | 0.852–0.956 |
| **Texture only (illumination-invariant)** | **0.940** | **0.896–0.973** |

Texture beats the strongest probe by 0.225. Script 08 tests the magnification step:
with a fixed 64 px window instead, texture falls to 0.907. If the signal came from camera
distance, normalising magnification would lower the score, but it raises it.

The mean-colour probe sits well above chance, so the **colour** result is still partly
confounded. Report the texture result.

## Dataset

The images were captured on a smartphone during routine inspection: 169 files, reduced
to **159 unique images from 132 animals** after removing 10 byte-identical duplicates.

| Class | Images | Animals |
|---|---|---|
| Normal | 97 | 79 |
| Fascioliasis | 39 | 34 |
| Abscess | 18 | 16 |
| Telangiectasis | 3 | 1 |
| Cirrhosis | 1 | 1 |
| Fibrosis | 1 | 1 |

Three lesion types come from a single animal each, so the analysis is binary (normal vs
lesioned). Several images can show the same liver, so every split is grouped by animal
and sample sizes are counted in animals, not images.

The raw images are not included in this repository. To run the pipeline, place them in
this layout:

```
Project Data/
├── normal_liver/
└── abnormal_liver/
    ├── Liver Abscesses/
    ├── Liver Fascioliasis/
    ├── LIVER Telangiectasis/
    ├── LIVER CIRRHOSIS/
    └── LIVER FIBROSIS/
```

## Installation

Tested on Python 3.14. The pipeline runs on CPU; a GPU only speeds up fine-tuning.

```bash
conda create -n analysis python=3.14
conda activate analysis
pip install -r requirements.txt
```

The first deep-learning run downloads ImageNet weights through torchvision.

## Running the pipeline

The scripts are numbered and share state through `outputs/manifest.csv`, so run them in
order from `scripts/`:

```bash
cd scripts

python 01_build_manifest.py           # scan, deduplicate, assign animal IDs
python 02_harmonise_and_probe.py      # harmonise images, run leakage probes
python 03_classical_baseline.py       # SVM / Random Forest / logistic regression
python 04_confound_diagnostics.py     # is the signal in the lesion or the photo?
python 05_deep_learning.py            # frozen backbones (~3 min on CPU)
python 05_deep_learning.py --finetune # full 3×2 grid (hours on CPU)
python 06_gradcam_occlusion.py        # where does the model look?
python 07_patch_analysis.py           # tissue-only analysis (headline result)
python 08_scale_ablation.py           # does magnification normalisation help?
```

[`Liver_Lesion_Analysis.ipynb`](Liver_Lesion_Analysis.ipynb) runs the same steps with
commentary. Metrics, out-of-fold predictions and figures are written to
[`outputs/`](outputs/).

The thesis chapters and technical report are generated from `outputs/` and are not
edited by hand. To rebuild them after a pipeline change:

```bash
python make_thesis_docx.py   # Liver_Lesion_Thesis_Methods_Results_Discussion.docx
python make_report_docx.py   # Liver_Lesion_Analysis_Report.docx
```

## Repository layout

```
Project Data/      raw images (not tracked; see Dataset)
data_harmonised/   generated: decoded, rotated, cropped, lossless PNG
src/liver/         pipeline modules
scripts/           numbered entry points and document generators
outputs/           manifest, metrics, predictions, diagnostics, figures
v0_archive/        original notebook, kept for before/after comparison
```

| Module | Responsibility |
|---|---|
| `manifest.py` | Scan, MD5-deduplicate, assign animal IDs from filenames and perceptual hashing |
| `harmonise.py` | Decode, EXIF-rotate, aspect-safe crop, save as lossless PNG |
| `probes.py` | Negative controls: capture metadata, compression artefacts, content statistics |
| `splits.py` | Nested `StratifiedGroupKFold`: outer folds score, inner folds select |
| `features.py` | Colour histograms (RGB + HSV) and GLCM texture |
| `models.py` | MobileNetV2, EfficientNet-B0 and ResNet18, frozen or fine-tuned |
| `train.py` | Per-fold training with early stopping on the inner split only |
| `evaluate.py` | Metrics, threshold selection, pooled out-of-fold AUC with animal-level bootstrap CI |
| `gradcam.py` | Grad-CAM, border-attention share, sliding-patch occlusion |
| `segment.py` | Colour-cluster liver segmentation and organ-relative patch sampling |

Harmonised images are saved as PNG on purpose. Re-encoding to JPEG was tested and
rejected: the two classes arrived at different original JPEG qualities, and double
compression made the block-grid artefact more class-predictive (probe AUC 0.61 → 0.98).
See [`src/liver/config.py`](src/liver/config.py).

## Acceptance gates for new data

Run scripts 02 and 04 on every new batch of images **before** training a model. If these
gates fail, adding more images will not fix the problem.

| Probe | Target | Current data |
|---|---|---|
| Capture metadata | ≈ 0.50 | 0.500 ✅ |
| Compression artefact | < 0.65 | 0.608 ✅ |
| Border ring only | < 0.65 | 0.971 ❌ |
| 8×8 thumbnail | < 0.70 | 0.989 ❌ |
| Grad-CAM border share | < 0.64 | 0.830 ❌ |

## Collection protocol

Only re-collection makes whole-image classification valid. The rule: someone shown only
the edges of the frame should not be able to tell a normal liver from a lesioned one
photographed minutes apart.

- **Photograph both classes in every session**, on the same plain inspection surface.
- **Take one standard shot per liver**: whole organ, fixed distance, same orientation.
  Close-ups are extra frames, never the only frame.
- **Use the same device and settings throughout** and copy files straight off the phone.
  The uniform 810×1080 size of the current normal images is a messaging-app export
  signature, and it is why file metadata alone predicted the class perfectly.
- **Record an animal ID for every image**, so grouping never has to be reconstructed
  from image similarity.

## Limitations

- One abattoir, one collection period, 132 animals.
- Liver segmentation is classical colour clustering, so some patches may include
  non-liver tissue.
- Binary only: too few animals per lesion type for multi-class classification.
- The colour pathway is still partly confounded (mean-colour probe AUC 0.715).
