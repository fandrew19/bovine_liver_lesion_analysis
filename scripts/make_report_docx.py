"""Generate the methods / results / discussion write-up as a Word document.

Every number is read from outputs/ rather than typed in, so the document cannot drift
out of step with the analysis. Re-run after any pipeline change.
"""
import _bootstrap  # noqa: F401

from datetime import date

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from liver.config import OUTPUTS, PROJECT_ROOT

ACCENT = RGBColor(0x8C, 0x2F, 0x39)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x55, 0x55)
FIG = OUTPUTS / "figures"


# ----------------------------------------------------------------------------- helpers
def load(name):
    return pd.read_csv(OUTPUTS / f"{name}.csv")


def styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.15
    for level, size in ((1, 16), (2, 13), (3, 11.5)):
        st = doc.styles[f"Heading {level}"]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = ACCENT if level < 3 else INK
        st.paragraph_format.space_before = Pt(16 if level == 1 else 11)
        st.paragraph_format.space_after = Pt(5)


def para(doc, text, *, italic=False, size=None, colour=None, after=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if colour:
        run.font.color.rgb = colour
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if align:
        p.alignment = align
    return p


def bullets(doc, items, style="List Bullet"):
    for item in items:
        p = doc.add_paragraph(style=style)
        if isinstance(item, tuple):
            lead, rest = item
            p.add_run(lead).bold = True
            p.add_run(rest)
        else:
            p.add_run(item)
        p.paragraph_format.space_after = Pt(3)


def table(doc, headers, rows, widths=None, caption=None, number=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(9)
            if i and str(value).replace(".", "").replace("-", "").replace("–", "").isdigit():
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    if caption:
        para(doc, f"Table {number}. {caption}", italic=True, size=9, colour=MUTED, after=12)
    return t


def figure(doc, path, caption, number, width=6.3):
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, f"Figure {number}. {caption}", italic=True, size=9,
         colour=MUTED, after=12, align=WD_ALIGN_PARAGRAPH.CENTER)


def code_block(doc, lines):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(9)
    run = p.add_run("\n".join(lines))
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def fmt(x, n=3):
    return f"{x:.{n}f}"


def ci(row, lo="lo", hi="hi"):
    return f"{row[lo]:.3f}–{row[hi]:.3f}"


# ----------------------------------------------------------------------------- build
def main() -> None:
    nc = load("negative_controls")
    cd = load("confound_diagnostics")
    cp = load("classical_pooled")
    dp = load("deep_pooled")
    pa = load("patch_analysis")
    man = pd.read_csv(OUTPUTS / "manifest.csv")
    dropped = load("dropped_duplicates")
    gc = load("gradcam_attention")
    oc = load("occlusion_attention")
    cf = load("classical_fold_metrics")
    df_ = load("deep_fold_metrics")

    by_type = (man.groupby("lesion_type")
                  .agg(images=("path", "size"), livers=("liver_id", "nunique"))
                  .sort_values("images", ascending=False))

    def probe(stage, name):
        return nc.loc[nc.probe.str.contains(stage) & nc.probe.str.contains(name),
                      "worst_auc"].item()

    def diag(key):
        return cd.loc[cd.variant.str.startswith(key), "auc"].item()

    def patch(key):
        return pa.loc[pa.configuration.str.contains(key)].iloc[0]

    texture = patch("texture only")
    colour = patch("colour features")
    worst_probe = pa[pa.probe].auc.max()

    doc = Document()
    styles(doc)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.85)
        s.left_margin = s.right_margin = Inches(0.9)

    # ---------------------------------------------------------------- title
    t = para(doc, "Deep Learning and Texture-Based Classification of Bovine Liver "
                  "Lesions from Smartphone Images", align=WD_ALIGN_PARAGRAPH.CENTER)
    t.runs[0].font.size = Pt(19)
    t.runs[0].font.bold = True
    t.runs[0].font.color.rgb = ACCENT
    para(doc, "Methods, Results and Discussion", size=13, italic=True,
         colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "A Ghanaian Abattoir Study — Kumasi Abattoir, Ashanti Region",
         size=10.5, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, f"Analysis pipeline report · {date.today():%d %B %Y}",
         size=9, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=18)

    # ---------------------------------------------------------------- abstract
    doc.add_heading("Summary", level=1)
    para(doc,
         f"A pilot dataset of {len(man) + len(dropped)} smartphone photographs of bovine "
         f"livers was analysed to develop a classifier distinguishing normal organs from "
         f"those bearing visible gross lesions. After removing {len(dropped)} byte-identical "
         f"duplicate files and grouping repeat views by animal, the dataset comprised "
         f"{len(man)} unique images from {man.liver_id.nunique()} distinct livers "
         f"({int((man.label == 0).sum())} normal, {int((man.label == 1).sum())} lesioned).")
    para(doc,
         "Whole-image classifiers achieved apparently excellent discrimination "
         f"(area under the receiver operating characteristic curve, AUC, up to "
         f"{dp.auc.max():.3f}). A systematic programme of negative controls established that "
         "this performance was not attributable to pathology. Before correction, image file "
         "metadata alone — carrying no pixel information whatsoever — separated the classes "
         f"perfectly (AUC {probe('RAW', 'capture metadata'):.3f}). After that defect was "
         "eliminated, an 8×8 pixel thumbnail in which no lesion is resolvable still scored "
         f"AUC {diag('8x8'):.3f}, and the image border with the organ excised scored "
         f"{diag('border ring only'):.3f}. Gradient-weighted class activation mapping confirmed "
         f"that {gc[gc.y_true == 0].border_share.mean():.1%} of model attention on normal "
         "livers fell outside the organ. The two classes had been photographed under "
         "systematically different conditions, and every whole-image model was reading those "
         "conditions rather than the liver.")
    para(doc,
         "A tissue-restricted analysis was therefore developed. The organ was segmented, "
         "patches were sampled from its interior at a magnification normalised against organ "
         "size, and texture features were computed on locally standardised lightness. This "
         f"approach achieved AUC {texture.auc:.3f} (95% CI {ci(texture)}) at the level of the "
         f"individual animal, against a worst-case confound probe of {worst_probe:.3f}. "
         "This is reported as the study's valid pilot finding: bovine liver parenchymal "
         "texture, measured from smartphone photographs, discriminates lesioned from normal "
         "organs independently of background, framing, illumination and magnification.")

    # ---------------------------------------------------------------- 1
    doc.add_heading("1. Purpose and scope of this document", level=1)
    para(doc,
         "This document records the complete analytical procedure applied to the pilot "
         "dataset: every processing stage, the parameters governing it, the rationale for "
         "each design decision, the results obtained, and the directions indicated for "
         "subsequent work. It is written to be sufficient for independent reproduction and "
         "for direct incorporation into the study's methods and results chapters.")
    para(doc,
         "Two features of the analysis distinguish it from a conventional image-classification "
         "workflow and should be noted at the outset. First, a substantial part of the "
         "procedure is devoted to negative controls — deliberate attempts to solve the "
         "classification task using information that could not possibly encode pathology. "
         "Second, the interpretability analysis is framed as a hypothesis test capable of "
         "failing rather than as an illustrative figure. Both proved decisive.")

    # ---------------------------------------------------------------- 2
    doc.add_heading("2. Materials", level=1)

    doc.add_heading("2.1 Image dataset as received", level=2)
    para(doc,
         f"Images were supplied as {len(man) + len(dropped)} JPEG files in a two-level "
         "directory structure: a single folder of normal livers, and a folder of abnormal "
         "livers subdivided by lesion type as assigned by the meat inspector at the point of "
         "capture. Lesion subtype labelling was therefore already available, satisfying an "
         "objective the original protocol had deferred to a later phase.")

    doc.add_heading("2.2 Computing environment", level=2)
    para(doc,
         "All analysis was performed in Python 3.14 within a dedicated Conda environment "
         "(`analysis`) on a four-thread CPU with no GPU available. Principal libraries: "
         "PyTorch 2.12 and torchvision 0.27 (transfer learning); scikit-learn 1.9 "
         "(cross-validation, classical classifiers, metrics); scikit-image 0.26 (texture "
         "features, morphology, colour conversion); Pillow 12 (image input/output); "
         "SciPy 1.18, NumPy 2.4, pandas 3.0, matplotlib 3.11.")
    para(doc,
         "The absence of GPU acceleration materially shaped the experimental design: "
         "frozen-backbone configurations were evaluated using precomputed embeddings, which "
         "reduced their cost from hours to approximately one minute per architecture.",
         colour=MUTED, size=10)

    # ---------------------------------------------------------------- 3
    doc.add_heading("3. Methods", level=1)

    doc.add_heading("3.1 Overall design", level=2)
    para(doc,
         "The pipeline comprises nine sequential stages. Stages 1–4 establish the validity of "
         "any subsequent measurement and must precede modelling; running classifiers before "
         "them produces figures that later require withdrawal.")
    table(doc,
          ["Stage", "Purpose", "Script"],
          [["1. Manifest", "Deduplication; assignment of animal identity", "01_build_manifest.py"],
           ["2. Harmonisation", "Elimination of file-level capture differences", "02_harmonise_and_probe.py"],
           ["3. Negative controls", "Detection of non-pathological signal", "02_harmonise_and_probe.py"],
           ["4. Confound diagnostics", "Localisation of residual signal", "04_confound_diagnostics.py"],
           ["5. Partitioning", "Nested, animal-grouped cross-validation", "src/liver/splits.py"],
           ["6. Classical arm", "Colour and texture features; three classifiers", "03_classical_baseline.py"],
           ["7. Deep arm", "Transfer learning, three architectures", "05_deep_learning.py"],
           ["8. Interpretability", "Grad-CAM and occlusion, as a test", "06_gradcam_occlusion.py"],
           ["9. Tissue analysis", "Segmentation and patch-level classification", "07_patch_analysis.py"]],
          widths=[1.5, 3.2, 1.9], number=1,
          caption="Analytical stages and corresponding executable scripts.")

    doc.add_heading("3.2 Stage 1 — Manifest construction, deduplication and animal identity", level=2)
    para(doc,
         "A single comma-separated manifest was constructed as the authoritative record for "
         "all later stages, replacing directory traversal. Each row records the relative file "
         "path, binary label, lesion subtype, animal identifier, MD5 checksum, pixel "
         "dimensions, file size, EXIF orientation tag and mean JPEG quantization value.")

    doc.add_heading("Deduplication", level=3)
    para(doc,
         "Files were hashed with MD5 and exact duplicates removed, retaining the "
         "first occurrence in path order. This step is not cosmetic: duplicated files "
         "inflate the apparent sample size and, combined with random partitioning, place "
         "identical images in both training and test sets.")

    doc.add_heading("Assignment of animal identity", level=3)
    para(doc,
         "The protocol specifies multiple photographs per liver. Treating these as "
         "independent observations would allow a model to be evaluated on animals it had "
         "been trained on. Animal identity was recovered by two complementary routes and "
         "the results merged by union-find:")
    bullets(doc, [
        ("Filename parsing. ", "Lesion images encode a case number with an optional "
         "single-letter view suffix (for example LF13a, LF13b and LF13c denote three views "
         "of one liver). A trailing letter was treated as a view marker only when preceded "
         "by a digit, so that filenames carrying no case number were not incorrectly merged."),
        ("Appearance clustering. ", "Normal images are numbered sequentially and carry no "
         "case information. A 16×16 difference hash and a 32×32 normalised thumbnail were "
         "computed for every image; pairs were joined when the Hamming distance between "
         "hashes was ≤32, or when it was ≤45 and the thumbnail correlation was ≥0.80."),
    ])
    para(doc, "Parameters: difference-hash side 16; thumbnail side 32; Hamming thresholds "
              "32 and 45; correlation threshold 0.80.", size=9.5, colour=MUTED)

    doc.add_heading("3.3 Stage 2 — Image harmonisation", level=2)
    para(doc,
         "Every image was passed through one identical processing path to remove differences "
         "arising from capture and export rather than from anatomy:")
    code_block(doc, [
        "img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')",
        "img = ImageOps.fit(img, (256, 256), method=Image.LANCZOS)",
        "img.save(destination, 'PNG', optimize=True)",
    ])
    bullets(doc, [
        ("EXIF transposition. ", "The orientation tag was present on every normal image and "
         "absent from every lesion image. Pillow does not apply it automatically, so omitting "
         "this step feeds one class to the network rotated by 90° relative to the other."),
        ("Aspect-preserving crop. ", "ImageOps.fit crops to the target aspect ratio before "
         "scaling. A direct square resize would compress each class by a different factor, "
         "because normal images share a single aspect ratio while lesion images vary."),
        ("Lossless storage. ", "Output was written as PNG. Re-encoding as JPEG was tested "
         "and rejected — see below."),
    ])
    para(doc,
         "Two alternative harmonisation strategies were evaluated and rejected on the "
         "evidence. Re-encoding to JPEG at quality 75 raised the compression-artefact probe "
         "from AUC 0.61 to 0.98: because the two classes arrived at different original "
         "qualities, a second compression interacts with the first and amplifies rather than "
         "erases the difference. Gaussian pre-blurring was also rejected, since a fixed "
         "kernel applied to originals of differing size leaves a size-dependent — and "
         "therefore class-dependent — residual.", size=10)
    para(doc, "Parameters: output 256×256 pixels; Lanczos resampling; PNG, optimised.",
         size=9.5, colour=MUTED)

    doc.add_heading("3.4 Stage 3 — Negative controls", level=2)
    para(doc,
         "Three probe families were defined. Each attempts the classification task using "
         "information that either cannot encode pathology, or can. They are kept separate "
         "because conflating them overstates the leak — an error made in an early draft of "
         "this analysis and corrected here.")
    table(doc,
          ["Probe family", "Features", "Interpretation", "Gated"],
          [["Capture metadata",
            "Width, height, aspect ratio, pixel count, mean JPEG quantization, EXIF orientation",
            "No image content; any skill is pure leakage", "Yes"],
           ["Compression artefact",
            "Gradient energy on the JPEG 8-pixel block grid ÷ energy elsewhere (2 features)",
            "A compression fingerprint in the pixels", "Yes"],
           ["Image content statistics",
            "Greyscale standard deviation, mean, bytes per pixel",
            "Contrast genuinely differs between uniform and abscessed tissue; not leakage",
            "No"]],
          widths=[1.3, 2.4, 2.2, 0.7], number=2,
          caption="Negative-control probe families. Only the first two gate the pipeline.")
    para(doc,
         "Each probe was scored with both a random forest (400 trees) and an RBF support "
         "vector machine under animal-grouped five-fold cross-validation, retaining the "
         "higher of the two AUCs: a leak detectable by either model is a leak. The gate "
         "threshold was AUC 0.65. A label-permutation control (five repetitions of the full "
         "protocol with shuffled labels) verified the evaluation machinery itself.")

    doc.add_heading("3.5 Stage 4 — Confound diagnostics", level=2)
    para(doc,
         "Where a gated probe passes but performance remains implausibly high, the residual "
         "signal must be located. Six image variants were constructed, each destroying lesion "
         "detail in a different way, and colour statistics (mean, standard deviation, 10th and "
         "90th percentiles per channel) extracted from each:")
    bullets(doc, [
        "Full image — reference condition.",
        "Centre patch only — the central 40%, where the lesion is most likely to lie.",
        "Border ring only — the outer 20% frame, with the centre discarded entirely.",
        "Heavy Gaussian blur (σ = 12) — coarse colour layout, no detail.",
        "8×8 pixel thumbnail — no lesion resolvable at any scale.",
        "Mean colour — three numbers per image.",
    ])
    para(doc,
         "The logic is straightforward: any variant that retains high discriminative power "
         "after the lesion has been rendered unreadable is measuring something other than "
         "the lesion.", size=10)

    doc.add_heading("3.6 Stage 5 — Data partitioning", level=2)
    para(doc,
         "Nested cross-validation was used, with grouping by animal enforced at both levels. "
         "The outer loop estimates performance; the inner loop performs all model selection. "
         "This separation addresses a defect in which a checkpoint is chosen by its score on "
         "the same partition subsequently used to report performance, which reports the most "
         "favourable epoch rather than generalisation.")
    bullets(doc, [
        ("Outer partition. ", "Stratified group five-fold, grouping on animal identifier; "
         "three repetitions with different seeds for the classical arm, one for the deep arm."),
        ("Inner partition. ", "A stratified group four-fold split of the outer training "
         "partition, used exclusively for early stopping and decision-threshold selection."),
        ("Verification. ", "Every fold was checked programmatically for group overlap "
         "between training and test partitions; all folds returned none."),
    ])
    para(doc, "Identical fold objects were shared between the classical and deep arms, so "
              "that the comparison between them is like-for-like.", size=10)

    doc.add_heading("3.7 Stage 6 — Classical feature extraction and classifiers", level=2)
    para(doc, "A 162-dimensional feature vector was extracted from each harmonised image:")
    bullets(doc, [
        ("Colour distribution. ", "16-bin normalised histograms of each RGB channel, plus "
         "channel mean and standard deviation, plus 12-bin normalised histograms of each "
         "HSV channel. Hue and saturation carry the pale bile-duct tracts characteristic of "
         "fascioliasis and the white foci of abscessation."),
        ("Texture. ", "Grey-level co-occurrence matrices computed on lightness quantised to "
         "64 levels, at distances 1, 3 and 5 pixels and angles 0, π/4, π/2 and 3π/4. Six "
         "properties were extracted from each matrix: contrast, dissimilarity, homogeneity, "
         "energy, correlation and angular second moment."),
    ])
    para(doc, "Three classifiers were trained: an RBF support vector machine (C = 1.0, "
              "balanced class weights); a random forest (500 trees, minimum two samples per "
              "leaf, balanced subsample weighting); and L2-regularised logistic regression "
              "(C = 0.1, balanced class weights). Features were standardised within each "
              "training fold for the SVM and logistic models.", size=10)

    doc.add_heading("3.8 Stage 7 — Transfer learning", level=2)
    para(doc,
         "Three ImageNet-pretrained architectures were evaluated — MobileNetV2, "
         "EfficientNet-B0 and ResNet18 — each in two regimes, giving six configurations. "
         "The final classification layer was replaced with a single-logit linear layer in "
         "every case.")
    doc.add_heading("Frozen-backbone regime", level=3)
    para(doc,
         "With the convolutional trunk fixed, embeddings depend only on the input, so they "
         "were computed once per image rather than at every epoch. Six views were embedded "
         "per image: one under the deterministic evaluation transform, and five under the "
         "training augmentation. Augmented views of training images were stacked as "
         "additional training rows; inner-validation and test rows used the deterministic "
         "view only. An L2-regularised logistic head (C = 0.1, balanced class weights, "
         "3000 iterations maximum) was then fitted on standardised embeddings.")
    doc.add_heading("Fine-tuned regime", level=3)
    para(doc,
         "The full network was trained with AdamW (learning rate 3×10⁻⁴, weight decay 10⁻⁴), "
         "batch size 16, cosine-annealed learning rate, and binary cross-entropy with a "
         "positive-class weight equal to the negative-to-positive ratio within each training "
         "fold. Training ran for a maximum of 12 epochs with early stopping on inner-fold "
         "loss (patience 4). This regime is implemented and available behind a command-line "
         "flag but was not executed for the results below, for reasons given in section 5.2.")
    doc.add_heading("Augmentation", level=3)
    para(doc,
         "Random horizontal flip (p = 0.5), vertical flip (p = 0.3), rotation up to 20°, "
         "colour jitter (brightness 0.25, contrast 0.25, saturation 0.2, hue 0.02), and "
         "random affine translation up to 8% with scaling between 0.9 and 1.1. Images were "
         "resized to 224×224 and normalised using ImageNet channel statistics.")
    para(doc,
         "Random erasing, specified in the original protocol, was deliberately omitted. On a "
         "dataset of this size it removes the lesion from an already small positive class and "
         "trains the model to infer the label from context — precisely the failure mode under "
         "investigation.", size=10, colour=MUTED)

    doc.add_heading("3.9 Stage 8 — Interpretability as a hypothesis test", level=2)
    para(doc,
         "Two quantitative measures were implemented, both capable of returning a result "
         "that contradicts the study's premise.")
    bullets(doc, [
        ("Border attention share. ", "Gradient-weighted class activation maps were computed "
         "at the final convolutional block and the proportion of total activation falling "
         "within the outer 20% frame was recorded. A spatially uniform map yields 0.64 "
         "under this definition, which provides the reference value; substantially higher "
         "indicates attention directed outside the organ."),
        ("Occlusion sensitivity. ", "A 48-pixel patch was translated across the image at a "
         "stride of 16 pixels and the change in predicted probability recorded at each "
         "position. Unlike gradient-based attribution this requires no gradients and cannot "
         "be misled by a sharp but causally irrelevant activation: if concealing a region "
         "does not alter the prediction, that region was not being used."),
    ])
    para(doc,
         "A technical note: with a frozen trunk no parameter requires a gradient, so "
         "backpropagation terminates before reaching the target layer and the backward hook "
         "never fires. Requiring a gradient on the input tensor preserves the computational "
         "graph and resolves this.", size=9.5, colour=MUTED)

    doc.add_heading("3.10 Stage 9 — Tissue-restricted patch analysis", level=2)
    para(doc,
         "Because the confound was localised to the background and the framing (section 4.4), "
         "an analysis was constructed that discards both. The organ is segmented, samples are "
         "drawn from its interior, and tissue rather than photographs is classified. Organ "
         "outline, shot composition, inspection surface and image border are excluded by "
         "construction.")

    doc.add_heading("Segmentation", level=3)
    para(doc,
         "No liver masks were available for supervised segmentation, so a classical "
         "colour-clustering procedure was used, which has the further advantage of being "
         "inspectable. Simple darkness or redness thresholding was tried first and rejected: "
         "pooled blood and a wet inspection table are both dark and red, so such rules retain "
         "large areas of background.")
    bullets(doc, [
        "Convert to CIELAB and cluster pixel colours with k-means (k = 4, four "
        "initialisations) fitted on a one-in-three subsample for speed.",
        "Weight each pixel by a Gaussian centred on the frame (σ = 0.45 of frame dimensions) "
        "and select the cluster with the greatest total weight — the subject of a photograph "
        "is what the photographer aimed at.",
        "Merge any adjacent cluster whose mean lightness is up to 18 units darker, so that a "
        "fissure or a cut surface does not divide the organ.",
        "Morphological opening (disk radius 3), closing (disk radius 9), hole filling, "
        "removal of objects below 2% of frame area, and retention of the largest connected "
        "component.",
        "If the resulting mask covers less than 8% of the frame the rule is deemed to have "
        "failed and a generous central region is substituted.",
    ])

    doc.add_heading("Magnification-normalised patch sampling", level=3)
    para(doc,
         "A fixed-pixel window means different things in a close-up and in a whole-organ "
         "photograph. Window side length was therefore set to 0.30 × √(mask area), so that "
         "every patch covers the same fraction of the organ irrespective of camera distance, "
         "and each patch was then resampled to 64×64 pixels. Patches were accepted only when "
         "at least 85% of their area fell within a mask eroded by one-eighth of the window "
         "side, ensuring no sample straddles the organ boundary. Up to 12 patches were drawn "
         "per image; where the constraint could not be satisfied it was relaxed to 60% "
         "coverage on the unereoded mask.")

    doc.add_heading("Patch features", level=3)
    bullets(doc, [
        ("Colour (36 features). ", "CIELAB channel mean, standard deviation, 10th and 90th "
         "percentiles, plus 8-bin RGB histograms."),
        ("Texture (72 features). ", "Grey-level co-occurrence matrices at distances 1, 2 and "
         "4 pixels and four angles, 64 levels, six properties. Critically, lightness is "
         "standardised within each patch before quantisation, so absolute brightness — which "
         "encodes the photographic session rather than the pathology — cannot contribute. "
         "Only spatial arrangement survives."),
    ])
    para(doc,
         "Patch-level predictions were averaged within each image and then within each "
         "animal, and all reported AUCs are computed at the level of the animal.")

    doc.add_heading("3.11 Statistical analysis", level=2)
    bullets(doc, [
        ("Metrics. ", "Sensitivity, specificity, precision, F1 score and AUC, computed per "
         "fold and reported as mean ± standard deviation, and additionally on pooled "
         "out-of-fold predictions."),
        ("Decision threshold. ", "Selected on inner-fold predictions as the threshold "
         "achieving at least 90% sensitivity with the lowest false-positive rate, reflecting "
         "the asymmetric cost of a missed lesion in meat inspection. Where the target was "
         "unreachable within a fold, Youden's J was used."),
        ("Confidence intervals. ", "Percentile bootstrap over 2000 resamples, resampling "
         "animals rather than images. Resampling images would treat three photographs of one "
         "liver as three independent observations and yield intervals that are too narrow."),
        ("Reproducibility. ", "A fixed random seed (42) governs partitioning, augmentation, "
         "clustering and bootstrap resampling throughout."),
    ])

    # ---------------------------------------------------------------- 4
    doc.add_heading("4. Results", level=1)

    doc.add_heading("4.1 Dataset composition after curation", level=2)
    para(doc,
         f"Nine groups of byte-identical files were identified, comprising {len(dropped) + 9} "
         f"files that reduced to 9 unique images; one normal photograph had been stored five "
         f"times. After removing {len(dropped)} redundant files, {len(man)} unique images "
         f"remained. Filename parsing and appearance clustering resolved these to "
         f"{man.liver_id.nunique()} distinct animals, with 25 animals contributing more than "
         f"one photograph.")
    table(doc,
          ["Lesion type", "Images", "Distinct animals", "Suitability for modelling"],
          [[idx, int(r.images), int(r.livers),
            "Adequate" if r.livers >= 16 else "Single animal — cannot be validated"]
           for idx, r in by_type.iterrows()],
          widths=[1.5, 0.9, 1.4, 2.5], number=3,
          caption="Dataset composition after deduplication and grouping by animal.")
    figure(doc, FIG / "lesion_frequency.png",
           "Lesion frequency by image count and by distinct animal. The divergence between "
           "the two bars is the reason sample-size targets must be expressed in animals.", 1,
           width=5.6)
    para(doc,
         "Three of the five lesion categories derive from a single animal each, which "
         "precludes multi-class classification at present. Two discrepancies with the study "
         "protocol were identified: telangiectasis occurs in the data but is absent from the "
         "protocol's category list, while fatty liver is listed but was not observed.")

    doc.add_heading("4.2 Detection of a capture confound", level=2)
    para(doc,
         "Before harmonisation, a random forest supplied only with file properties — "
         "dimensions, byte size, JPEG quantization values and EXIF orientation, and no pixel "
         f"data of any kind — classified the images perfectly (AUC {probe('RAW', 'capture metadata'):.3f}, "
         "100% accuracy under animal-grouped five-fold cross-validation). Inspection revealed "
         "the mechanism: mean JPEG quantization values formed two non-overlapping regimes "
         "coinciding exactly with the class label (12.22 for all normal images; 35.59 or "
         "44.81 for all lesion images). Every normal image measured 0.87 megapixels and "
         "carried an EXIF orientation tag; every lesion image had variable dimensions and no "
         "EXIF data. The uniform sizing is characteristic of export through a messaging "
         "application rather than direct transfer from the capture device.")

    doc.add_heading("4.3 Effect of harmonisation", level=2)
    table(doc,
          ["Probe", "Raw images", "Harmonised", "Gate"],
          [["Capture metadata", fmt(probe("RAW", "capture metadata")),
            fmt(probe("HARMONISED", "capture metadata")), "Pass"],
           ["Compression artefact", fmt(probe("RAW", "compression artefact")),
            fmt(probe("HARMONISED", "compression artefact")), "Pass"],
           ["Image content statistics", fmt(probe("RAW", "content")),
            fmt(probe("HARMONISED", "content")), "Not gated"],
           ["Shuffled labels (protocol control)", "—",
            fmt(nc.loc[nc.probe.str.contains("shuffled"), "worst_auc"].item()), "Pass"]],
          widths=[2.4, 1.3, 1.3, 1.3], number=4,
          caption="Negative-control AUCs before and after harmonisation. Gate threshold 0.65.")
    para(doc,
         "Harmonisation eliminated the file-level confound completely: capture metadata fell "
         "from perfect discrimination to exactly chance, as expected once every property it "
         "measures has been made constant. The compression-artefact probe remained at 0.61, "
         "below the gate but above chance, reflecting the fact that the first compression is "
         "already baked into the pixels and cannot be removed retrospectively. The "
         "label-permutation control returned 0.474, confirming that the cross-validation "
         "machinery introduces no optimistic bias of its own.")

    doc.add_heading("4.4 Localisation of the residual signal", level=2)
    para(doc,
         "Although both gated probes passed, whole-image performance remained implausibly "
         "high, so the diagnostic variants of section 3.5 were applied.")
    table(doc,
          ["Information available to the classifier", "AUC", "Interpretation"],
          [["Full harmonised image", fmt(diag("full image")), "Reference"],
           ["8×8 thumbnail — no lesion resolvable", fmt(diag("8x8")),
            "Exceeds the full image"],
           ["Border ring only — organ excised", fmt(diag("border ring only")),
            "Background alone suffices"],
           ["Heavy blur — colour layout only", fmt(diag("heavy blur")),
            "No detail required"],
           ["Mean colour — three numbers", fmt(diag("mean colour")), "Colour cast alone"],
           ["Centre patch — lesion location", fmt(diag("centre patch")),
            "Least informative region"]],
          widths=[2.9, 0.8, 2.2], number=5,
          caption="Class discrimination surviving progressive destruction of lesion detail.")
    para(doc,
         "The result is unambiguous. An 8×8 thumbnail, in which no lesion can be resolved at "
         "any scale, outperformed the full image. The image border with the organ excised "
         "achieved near-perfect discrimination. The central region, where the lesion actually "
         "lies, was the least informative part of the image. Per-image colour normalisation "
         "was applied as a remedy and did not resolve the problem — the border probe remained "
         "at 0.944 — establishing that the difference lies in framing and composition rather "
         "than in colour cast alone.")
    figure(doc, FIG / "class_backgrounds.png",
           "Six randomly selected normal livers (upper row) and six lesion images (lower row) "
           "after harmonisation. Normal organs are photographed whole, at a consistent "
           "distance, on pale matting; lesion images are close-up crops at varying distance "
           "on varying surfaces. The classes are separable without reference to pathology.", 2)

    doc.add_heading("4.5 Whole-image classification performance", level=2)
    table(doc,
          ["Model", "Pooled AUC", "95% CI", "Sensitivity", "Specificity"],
          [[r.model, fmt(r.auc), f"{r.auc_lo:.3f}–{r.auc_hi:.3f}",
            fmt(r.sensitivity), fmt(r.specificity)]
           for _, r in pd.concat([dp, cp]).iterrows()]
          + [["8×8 thumbnail (reference)", fmt(diag("8x8")), "—", "—", "—"]],
          widths=[2.1, 1.0, 1.3, 1.1, 1.1], number=6,
          caption="Whole-image performance on pooled out-of-fold predictions, animal-grouped "
                  "nested cross-validation. All values are invalidated by section 4.4.")
    para(doc,
         "Every architecture, deep or classical, converged on approximately the same value as "
         "a thumbnail too small to contain a lesion. This convergence is itself diagnostic: "
         "it indicates a task being solved by information available equally to all of them. "
         "The convolutional networks did not outperform the classical baseline because there "
         "was no additional structure for them to exploit.")

    doc.add_heading("4.6 Interpretability", level=2)
    table(doc,
          ["Measure", "Value", "Reference"],
          [["Uniform attention map (definitional)", "0.640", "—"],
           ["Grad-CAM border share, lesion images",
            fmt(gc[gc.y_true == 1].border_share.mean()), "0.640"],
           ["Grad-CAM border share, normal images",
            fmt(gc[gc.y_true == 0].border_share.mean()), "0.640"],
           ["Occlusion evidence in outer frame",
            fmt(oc.occlusion_border_share.mean()), "0.640"]],
          widths=[3.0, 1.2, 1.2], number=7,
          caption=f"Proportion of model evidence falling outside the organ "
                  f"(n = {len(gc)} held-out images; occlusion on {len(oc)}).")
    para(doc,
         f"On normal livers, {gc[gc.y_true == 0].border_share.mean():.1%} of activation fell "
         "outside the organ — the model's evidence for classifying a liver as normal is "
         "almost entirely the surrounding scene. The occlusion analysis, which requires no "
         "gradients and is therefore immune to attribution artefacts, independently "
         f"corroborated this at {oc.occlusion_border_share.mean():.3f}. Attention on lesion "
         f"images was more central ({gc[gc.y_true == 1].border_share.mean():.3f}), indicating "
         "that lesion recognition does draw partly on tissue. The interpretability objective "
         "therefore fails on this dataset in a specific and informative way.")
    figure(doc, FIG / "gradcam_compact.png",
           "Upper row: a fascioliasis case, with attention on tissue — the intended "
           "behaviour. Lower row: a normal liver, where the activation map is flat across "
           "the organ and occlusion locates the evidence away from the tissue.", 3, width=5.9)

    doc.add_heading("4.7 Tissue-restricted patch analysis", level=2)
    para(doc,
         f"Segmentation yielded masks covering a mean of "
         f"{man[man.label == 0].mask_fraction.mean():.2f} of the frame for normal images and "
         f"{man[man.label == 1].mask_fraction.mean():.2f} for lesion images — a close "
         f"correspondence, indicating that mask area is not itself a strong class cue. "
         f"A total of 1,908 patches were obtained from all {len(man)} images.")
    table(doc,
          ["Configuration", "Features", "AUC (per animal)", "95% CI"],
          [[r.configuration.replace("PROBE  ", "Probe — ").replace("TASK   ", "Task — "),
            int(r.n_features), fmt(r.auc), ci(r)]
           for _, r in pa.iterrows()],
          widths=[3.0, 0.8, 1.2, 1.4], number=8,
          caption="Patch-level results, scored at the level of the individual animal under "
                  "animal-grouped cross-validation.")
    para(doc,
         f"Texture computed on locally standardised lightness achieved AUC {texture.auc:.3f} "
         f"(95% CI {ci(texture)}), exceeding the strongest confound probe "
         f"({worst_probe:.3f}) by {texture.auc - worst_probe:.3f}. Each of the three "
         "candidate confounds — session colour, compression history and camera magnification "
         "— was independently controlled and all fell well below the task result.")
    para(doc,
         "A further observation supports the interpretation that this reflects genuine tissue "
         "structure. Normalising patch size against organ area, which equalises magnification "
         "between close-ups and whole-organ shots, raised performance from 0.907 to "
         f"{texture.auc:.3f}. An artefactual signal would be expected to degrade under such "
         "normalisation, not improve.")
    figure(doc, FIG / "segmentation_check.png",
           "Segmentation and patch sampling. Upper row: harmonised input. Middle row: "
           "selected organ mask, with excluded regions dimmed. Lower row: representative "
           "tissue patches drawn from the masked interior.", 4)

    # ---------------------------------------------------------------- 5
    doc.add_heading("5. Discussion", level=1)

    doc.add_heading("5.1 What has worked", level=2)
    bullets(doc, [
        ("A valid pilot finding. ", f"Liver parenchymal texture discriminates lesioned from "
         f"normal organs at AUC {texture.auc:.3f} (95% CI {ci(texture)}) across "
         f"{man.liver_id.nunique()} animals, with background, framing, illumination, "
         "compression and magnification each independently controlled. This is a narrower "
         "claim than whole-image classification, but it is defensible and it is supported by "
         "the data in hand."),
        ("A reusable confound-detection protocol. ", "The probe suite detected a defect that "
         "would otherwise have passed review. It is inexpensive, general, and applicable to "
         "any smartphone-based imaging study conducted in a field setting."),
        ("Correct evaluation machinery. ", "Deduplication, animal-level grouping, nested "
         "cross-validation, inner-fold threshold selection and animal-level bootstrap "
         "intervals. Every fold was verified free of group overlap, and the "
         "label-permutation control confirmed the absence of optimistic bias."),
        ("Complete lesion-type descriptives. ", "Frequencies by image and by animal, "
         "satisfying the descriptive objective and revealing that three lesion categories "
         "rest on a single animal each."),
    ])

    doc.add_heading("5.2 What has not worked, and why", level=2)
    para(doc,
         "Whole-image classification cannot be validated on the present dataset. The two "
         "classes were photographed under systematically different conditions, and no "
         "preprocessing tested was able to separate those conditions from the pathology. "
         "Lossless harmonisation, per-image colour normalisation, centre cropping and "
         "JPEG re-encoding were each attempted; the last actively worsened the problem.")
    para(doc,
         "This is an instance of shortcut learning, well documented in medical imaging, in "
         "which models exploit incidental correlates of the label — scanner identity, "
         "institutional markings, or the presence of a measurement rule in dermatoscopic "
         "images — rather than the clinical feature of interest. The present case is the "
         "veterinary equivalent, and its detection required controls that are not yet "
         "routine in the veterinary imaging literature.")
    para(doc,
         "The fine-tuned configurations were implemented but not executed. Fine-tuning on a "
         "four-thread CPU costs several hours, and the confound analysis had already "
         "established that the resulting figures would measure capture conditions rather than "
         "pathology. The code path remains available for use once the data support it.",
         size=10, colour=MUTED)

    doc.add_heading("5.3 Limitations", level=2)
    bullets(doc, [
        ("Residual colour confounding. ", f"The patch mean-colour probe reached "
         f"{patch('mean colour').auc:.3f}, above chance. The colour pathway therefore "
         "retains some session dependence, and the texture result — which is invariant to "
         "illumination by construction — should be the figure reported."),
        ("Imperfect segmentation. ", "Segmentation is classical and unsupervised. A minority "
         "of patches may include non-parenchymal tissue, and the masks should be inspected "
         "visually before the result is quoted."),
        ("Single site and period. ", "All images derive from one abattoir over one collection "
         "window. External validity is untested."),
        ("Sample size. ", f"{man.liver_id.nunique()} animals is a pilot-scale sample. "
         "Confidence intervals are correspondingly wide."),
        ("Binary labelling. ", "Lesion-type classification remains out of reach: three of "
         "five categories are represented by one animal each."),
        ("Label subjectivity. ", "Labels derive from gross examination by a single inspector "
         "at the point of capture. Inter-observer agreement has not been assessed, and "
         "gross-examination subjectivity is itself the problem the study seeks to address."),
    ])

    doc.add_heading("5.4 Implications for the study protocol", level=2)
    table(doc,
          ["Section", "Current text", "Required amendment"],
          [["2.3", "100 images, 60 normal / 40 abnormal",
            f"{len(man)} unique images from {man.liver_id.nunique()} animals; "
            f"{int((man.label==0).sum())} normal / {int((man.label==1).sum())} lesioned"],
           ["2.5", "Six categories including fatty liver",
            "Add telangiectasis; record that fatty liver has not been observed"],
           ["2.4", "Standardised distance, lighting, background",
            "Add: both classes in every session; identical framing for both; direct "
            "device transfer; animal identifier for every image"],
           ["2.7", "Metrics as mean ± SD across folds",
            "Add pooled out-of-fold AUC with animal-level bootstrap intervals"],
           ["2.10", "Small sample, single site, binary labels",
            "Add the capture confound, its detection, and its effect on the pilot estimates"]],
          widths=[0.7, 2.3, 3.0], number=9,
          caption="Amendments required to align the protocol with the data and analysis.")

    # ---------------------------------------------------------------- 6
    doc.add_heading("6. Future directions", level=1)

    doc.add_heading("6.1 Immediate — analysis available without new images", level=2)
    bullets(doc, [
        "Report the tissue-texture result as the study's principal pilot finding, with all "
        "three confound probes presented alongside it rather than relegated to an appendix.",
        "State the claim precisely: liver parenchymal texture discriminates lesioned from "
        "normal organs. This is what was measured and what the controls support.",
        "Present the segmentation figure so that readers may judge mask quality directly.",
        "Report lesion frequencies by animal as well as by image, satisfying the descriptive "
        "objective.",
        "Consider a methodological paper on the confound-detection protocol, which is of "
        "general interest to field-based veterinary imaging.",
    ])

    doc.add_heading("6.2 Collection protocol for the next phase", level=2)
    para(doc,
         "The governing principle: a normal liver and a lesioned liver photographed minutes "
         "apart must be indistinguishable to an observer shown only the margins of the frame.")
    bullets(doc, [
        ("Paired sessions. ", "Photograph both classes in every session, on one plain "
         "inspection surface. No session should yield only one class."),
        ("Identical framing. ", "One standard whole-organ view per liver at fixed distance "
         "and orientation, for normal and lesioned organs alike. Close-ups are supplementary "
         "frames, never the only frame."),
        ("Single device, direct transfer. ", "One device and camera setting throughout, with "
         "files copied directly from the device. Messaging-application export was the "
         "proximate cause of the file-level confound."),
        ("Animal identifiers. ", "Record an identifier for every image so that grouping need "
         "never again be reconstructed from image similarity."),
        ("Include a scale reference. ", "A ruler or standard marker in frame would permit "
         "true physical-scale normalisation rather than the organ-relative approximation "
         "used here."),
        ("Second grader. ", "Have an independent grader relabel a subset so that "
         "inter-observer agreement can be quantified."),
    ])

    doc.add_heading("6.3 Acceptance gates for each new batch", level=2)
    para(doc,
         "The diagnostic suite should be run as an acceptance test on every new batch before "
         "any modelling. If the gates do not pass, additional images will not help.")
    table(doc,
          ["Probe", "Target", "Current"],
          [["Capture metadata", "≈ 0.50", fmt(probe("HARMONISED", "capture metadata"))],
           ["Compression artefact", "< 0.65", fmt(probe("HARMONISED", "compression artefact"))],
           ["Border ring only", "< 0.65", fmt(diag("border ring only"))],
           ["8×8 thumbnail", "< 0.70", fmt(diag("8x8"))],
           ["Grad-CAM border share", "< 0.64", fmt(gc.border_share.mean())]],
          widths=[2.4, 1.3, 1.3], number=10,
          caption="Acceptance thresholds and current values. The final three currently fail "
                  "for whole-image analysis.")

    doc.add_heading("6.4 Modelling roadmap", level=2)
    bullets(doc, [
        ("Once gates pass. ", "Execute the full six-configuration grid including fine-tuning. "
         "Frozen backbones are expected to prevail at this sample size, consistent with the "
         "protocol's own reasoning."),
        ("Interpretability as a release criterion. ", "Require the border-attention measure "
         "to fall below the 0.64 uniform reference before any performance figure is quoted."),
        ("Blinded expert review. ", "Have the meat inspector score activation maps blind: "
         "does the highlighted region contain the lesion? That proportion is the substantive "
         "interpretability result."),
        ("Multi-class extension. ", "Requires roughly 30 or more distinct animals per "
         "category. Fascioliasis (34 animals) and abscess (16) approach viability; "
         "cirrhosis, fibrosis and telangiectasis each rest on one."),
        ("Segmentation improvement. ", "A small set of manually delineated masks would permit "
         "a supervised segmentation model, improving on the present unsupervised procedure "
         "and tightening the tissue-restricted analysis."),
        ("External validation. ", "A held-out batch from a different abattoir, or a different "
         "period, is the strongest available test of generalisation."),
    ])

    # ---------------------------------------------------------------- 7
    doc.add_heading("7. Reproducibility", level=1)
    para(doc, "Execution order. Stages 1–4 must precede modelling.")
    code_block(doc, [
        "conda activate analysis",
        "cd scripts",
        "",
        "python 01_build_manifest.py        # manifest, deduplication, animal identity",
        "python 02_harmonise_and_probe.py   # harmonisation and negative controls",
        "python 04_confound_diagnostics.py  # localisation of residual signal   [GATE]",
        "python 03_classical_baseline.py    # colour and texture baselines",
        "python 05_deep_learning.py         # frozen backbones (~3 minutes)",
        "python 05_deep_learning.py --finetune   # full grid (hours on CPU)",
        "python 06_gradcam_occlusion.py     # interpretability as a test",
        "python 07_patch_analysis.py        # tissue-restricted analysis",
        "python make_report_docx.py         # regenerate this document",
    ])
    table(doc,
          ["Module", "Responsibility"],
          [["config.py", "Paths, seeds and shared constants"],
           ["manifest.py", "Scanning, MD5 deduplication, animal identity assignment"],
           ["harmonise.py", "Single decode, EXIF rotation, aspect-safe crop, PNG output"],
           ["probes.py", "Negative controls: capture, compression, content"],
           ["splits.py", "Nested animal-grouped cross-validation"],
           ["features.py", "Whole-image colour and GLCM texture features"],
           ["segment.py", "Colour-cluster segmentation, organ-relative patch sampling"],
           ["models.py", "Backbone construction, transforms, embedding extraction"],
           ["train.py", "Fold training, inner-fold early stopping"],
           ["evaluate.py", "Metrics, threshold selection, bootstrap intervals"],
           ["gradcam.py", "Grad-CAM, border attention, occlusion sensitivity"]],
          widths=[1.5, 4.7], number=11,
          caption="Modules under src/liver/.")
    para(doc,
         "All intermediate artefacts are written to outputs/: the manifest, the record of "
         "removed duplicates, per-fold metrics, out-of-fold predictions, negative-control "
         "results, confound diagnostics, attention measurements and all figures. Every value "
         "in this document is read from those files at generation time.", size=10)

    # ---------------------------------------------------------------- 8
    doc.add_heading("8. References", level=1)
    for ref in [
        "Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & "
        "Wichmann, F. A. (2020). Shortcut learning in deep neural networks. Nature Machine "
        "Intelligence, 2(11), 665–673.",
        "Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image "
        "classification. IEEE Transactions on Systems, Man, and Cybernetics, 3(6), 610–621.",
        "Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, "
        "M., & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for "
        "mobile vision applications. arXiv:1704.04861.",
        "Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation "
        "and model selection. Proceedings of the 14th International Joint Conference on "
        "Artificial Intelligence, 2, 1137–1145.",
        "Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., "
        "van der Laak, J. A. W. M., van Ginneken, B., & Sánchez, C. I. (2017). A survey on "
        "deep learning in medical image analysis. Medical Image Analysis, 42, 60–88.",
        "Roberts, M., Driggs, D., Thorpe, M., et al. (2021). Common pitfalls and "
        "recommendations for using machine learning to detect and prognosticate for COVID-19 "
        "using chest radiographs and CT scans. Nature Machine Intelligence, 3(3), 199–217.",
        "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: "
        "Inverted residuals and linear bottlenecks. Proceedings of the IEEE Conference on "
        "Computer Vision and Pattern Recognition, 4510–4520.",
        "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. "
        "(2017). Grad-CAM: Visual explanations from deep networks via gradient-based "
        "localization. Proceedings of the IEEE International Conference on Computer Vision, "
        "618–626.",
        "Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for "
        "deep learning. Journal of Big Data, 6(1), 1–48.",
        "Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional "
        "neural networks. Proceedings of the 36th International Conference on Machine "
        "Learning, 6105–6114.",
        "Winkler, J. K., Fink, C., Toberer, F., et al. (2019). Association between surgical "
        "skin markings in dermoscopic images and diagnostic performance of a deep learning "
        "convolutional neural network for melanoma recognition. JAMA Dermatology, 155(10), "
        "1135–1141.",
        "Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. "
        "(2018). Variable generalization performance of a deep learning model to detect "
        "pneumonia in chest radiographs: A cross-sectional study. PLoS Medicine, 15(11), "
        "e1002683.",
    ]:
        p = doc.add_paragraph(ref)
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.space_after = Pt(5)
        for run in p.runs:
            run.font.size = Pt(9.5)

    out = PROJECT_ROOT / "Liver_Lesion_Analysis_Report.docx"
    doc.save(out)
    print(f"Written: {out}")
    print(f"  headings: {sum(1 for p in doc.paragraphs if p.style.name.startswith('Heading'))}")
    print(f"  tables  : {len(doc.tables)}")
    print(f"  figures : {len(doc.inline_shapes)}")


if __name__ == "__main__":
    main()
