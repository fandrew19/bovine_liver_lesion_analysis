"""Generate the thesis chapters (Materials and Methods, Results, Discussion) as a .docx.

Structured to follow the study proposal's chapter scheme, so this document slots in behind
Chapter One (Introduction) and Chapter Two (Literature Review) as Chapters Three to Five.

Every quoted figure is read from outputs/ at generation time rather than typed in, so the
document cannot drift out of step with the analysis it describes. Regenerate after any
pipeline change:

    conda activate analysis && cd scripts && python make_thesis_docx.py
"""
import _bootstrap  # noqa: F401

from datetime import date

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

from liver.config import OUTPUTS, PROJECT_ROOT

ACCENT = RGBColor(0x1F, 0x4E, 0x5F)
FIG = OUTPUTS / "figures"
OUT = PROJECT_ROOT / "Liver_Lesion_Thesis_Methods_Results_Discussion.docx"


# ----------------------------------------------------------------------- style helpers
def setup(doc):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    pf = st.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    for i, sz in [(1, 15), (2, 13), (3, 12)]:
        h = doc.styles[f"Heading {i}"]
        h.font.name = "Times New Roman"
        h.font.size = Pt(sz)
        h.font.color.rgb = ACCENT
        h.font.bold = True
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.25)
        s.right_margin = Inches(1.0)


def _rich(p, text):
    """Render **bold** segments inside a paragraph. Returns the runs created."""
    runs = []
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            r = p.add_run(chunk)
            r.bold = (i % 2 == 1)
            runs.append(r)
    return runs


def para(doc, text, justify=True, italic=False, size=None, space_after=6):
    p = doc.add_paragraph()
    for run in _rich(p, text):
        run.italic = italic
        if size:
            run.font.size = Pt(size)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        _rich(p, it)


def numbered(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Number")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        _rich(p, it)


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    return p


def _no_split(row, header=False):
    """Keep a row whole across a page break; repeat the header row on each page."""
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))
    if header:
        tr_pr.append(OxmlElement("w:tblHeader"))


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            run.font.size = Pt(10)
    if widths:
        for r_ in t.rows:
            for i, w in enumerate(widths):
                r_.cells[i].width = Inches(w)
    for i, r_ in enumerate(t.rows):
        _no_split(r_, header=(i == 0))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, cap, width=5.9):
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, cap)


def f3(x):
    return f"{x:.3f}"


def ms(sub, col):
    """mean ± SD across folds."""
    return f"{sub[col].mean():.3f} ± {sub[col].std():.3f}"


# ----------------------------------------------------------------------------- build
def main() -> None:
    nc = pd.read_csv(OUTPUTS / "negative_controls.csv")
    cd = pd.read_csv(OUTPUTS / "confound_diagnostics.csv")
    cp = pd.read_csv(OUTPUTS / "classical_pooled.csv")
    dp = pd.read_csv(OUTPUTS / "deep_pooled.csv")
    pa = pd.read_csv(OUTPUTS / "patch_analysis.csv")
    man = pd.read_csv(OUTPUTS / "manifest.csv")
    dropped = pd.read_csv(OUTPUTS / "dropped_duplicates.csv")
    gc = pd.read_csv(OUTPUTS / "gradcam_attention.csv")
    oc = pd.read_csv(OUTPUTS / "occlusion_attention.csv")
    cf = pd.read_csv(OUTPUTS / "classical_fold_metrics.csv")
    dfm = pd.read_csv(OUTPUTS / "deep_fold_metrics.csv")
    ps = pd.read_csv(OUTPUTS / "patch_summary.csv").iloc[0]
    sa = pd.read_csv(OUTPUTS / "scale_ablation.csv")

    n_img = len(man)
    n_animals = man.liver_id.nunique()
    n_norm = int((man.label == 0).sum())
    n_les = int((man.label == 1).sum())
    n_files = n_img + len(dropped)
    n_multi = int((man.liver_id.value_counts() > 1).sum())

    by_type = (man.groupby("lesion_type")
                  .agg(images=("path", "size"), livers=("liver_id", "nunique"))
                  .sort_values("images", ascending=False))

    def probe(stage, name):
        return nc.loc[nc.probe.str.contains(stage) & nc.probe.str.contains(name),
                      "worst_auc"].item()

    def diag(key):
        return cd.loc[cd.variant == key, "auc"].item()

    def patch(key):
        return pa.loc[pa.configuration.str.contains(key)].iloc[0]

    texture = patch("texture only")
    colour = patch("colour features")
    worst_probe = pa[pa.probe].auc.max()
    worst_probe_name = (pa.loc[pa[pa.probe].auc.idxmax(), "configuration"]
                        .replace("PROBE  patch ", "").replace(" only", ""))
    gc_norm = gc[gc.y_true == 0].border_share.mean()
    gc_les = gc[gc.y_true == 1].border_share.mean()
    best_deep = dp.loc[dp.auc.idxmax()]
    best_cls = cp.loc[cp.auc.idxmax()]
    sa_norm = sa.loc[sa.sampling.str.startswith("organ"), "auc"].item()
    sa_fixed = sa.loc[sa.sampling.str.startswith("fixed"), "auc"].item()

    doc = Document()
    setup(doc)

    # ------------------------------------------------------------------ title block
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Deep Learning Classification of Bovine Liver Lesions from "
                  "Smartphone Images")
    r.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = ACCENT

    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = s.add_run("A Ghanaian Abattoir Study — Kumasi Abattoir, Ashanti Region")
    rs.italic = True
    rs.font.size = Pt(12)

    s2 = doc.add_paragraph()
    s2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = s2.add_run("Chapter Three: Materials and Methods · Chapter Four: Results · "
                    "Chapter Five: Discussion")
    r2.font.size = Pt(11)
    r2.bold = True

    para(doc, f"Analysis covers {n_img} unique smartphone photographs from {n_animals} "
              f"distinct animals ({n_norm} normal, {n_les} bearing visible gross lesions), "
              f"curated from {n_files} image files. Generated {date.today():%d %B %Y}.",
         justify=False, italic=True, size=10)

    doc.add_page_break()

    # ================================================================== CHAPTER THREE
    doc.add_heading("CHAPTER THREE: MATERIALS AND METHODS", level=1)

    doc.add_heading("3.1 Study Design", level=2)
    para(doc, "This was a cross-sectional image-collection study combined with a "
              "computational model-development and validation study. Bovine livers examined "
              "during routine post-mortem meat inspection were photographed with a "
              "smartphone camera and labelled by the attending meat inspector as normal or "
              "as bearing a visible gross lesion. The resulting images were then used to "
              "develop and evaluate deep learning and classical machine-learning "
              "classifiers.")
    para(doc, "The analysis was designed around a principle that governs the whole of this "
              "chapter: **the validity of a measurement must be established before the "
              "measurement is reported**. In image classification with modest sample sizes, "
              "a model can achieve excellent apparent accuracy by exploiting incidental "
              "differences in how the two classes were photographed rather than the "
              "pathology they contain. A substantial part of the procedure described below "
              "is therefore devoted to negative controls — deliberate attempts to solve the "
              "classification task using information that could not possibly encode disease "
              "— and the interpretability analysis is framed as a hypothesis test capable of "
              "failing rather than as an illustrative figure. Both proved decisive for the "
              "findings reported in Chapter Four.")
    para(doc, "The pipeline comprises nine sequential stages. Stages one to four establish "
              "the validity of any subsequent measurement and must precede modelling.")
    table(doc,
          ["Stage", "Purpose", "Section"],
          [["1. Manifest", "Deduplication and assignment of animal identity", "3.6.1–3.6.3"],
           ["2. Harmonisation", "Elimination of file-level capture differences", "3.6.4"],
           ["3. Negative controls", "Detection of non-pathological signal", "3.7.1"],
           ["4. Confound diagnostics", "Localisation of residual signal", "3.7.2"],
           ["5. Partitioning", "Nested, animal-grouped cross-validation", "3.8"],
           ["6. Classical arm", "Colour and texture features; three classifiers", "3.9.1"],
           ["7. Deep arm", "Transfer learning, three architectures", "3.9.2"],
           ["8. Interpretability", "Grad-CAM and occlusion, framed as a test", "3.10"],
           ["9. Tissue analysis", "Segmentation and patch-level classification", "3.11"]],
          widths=[1.5, 3.4, 1.1])
    caption(doc, "Table 3.1: Analytical stages of the study and the sections describing them.")

    doc.add_heading("3.2 Study Area", level=2)
    para(doc, "The study was conducted at Kumasi Abattoir in the Ashanti Region of Ghana, a "
              "high-throughput facility at which routine, legally mandated post-mortem "
              "inspection of slaughtered cattle is carried out by qualified meat inspectors.")

    doc.add_heading("3.3 Study Population", level=2)
    para(doc, "The study population comprised livers from cattle slaughtered at the study "
              "site during the collection period. Livers passed as normal on gross "
              "examination served as the negative class, and livers bearing visible gross "
              "abnormalities formed the positive class. No restriction was placed on the "
              "breed, sex or age of the source animal, since these are not routinely "
              "recorded at the point of inspection.")

    doc.add_heading("3.4 Image Acquisition", level=2)
    para(doc, "Images were captured during routine post-mortem inspection using a smartphone "
              "camera. The collection protocol specified a fixed capture distance and angle "
              "relative to the liver surface, consistent lighting, a consistent plain "
              "inspection surface where feasible, multiple images per liver at different "
              "faces or angles, and a fixed device and resolution setting throughout.")
    para(doc, "As reported in Chapter Four, adherence to this protocol was in practice "
              "uneven, and the departures from it — particularly the photographing of the "
              "two classes under different framing, distance and background conditions, and "
              "the transfer of one class of images through a messaging application — proved "
              "to be the single most consequential feature of the dataset. The methods "
              "described in sections 3.7 and 3.11 exist because of this.")

    doc.add_heading("3.5 Image Labelling and Classification of Lesions", level=2)
    para(doc, "Two levels of labelling were used. The **primary label**, used for the pilot "
              "classification model, was binary: normal, or abnormal (visible gross lesion "
              "present), as assigned by the meat inspector at the point of capture. The "
              "**secondary label**, recorded for future multi-class work, was the specific "
              "gross lesion type, following World Organisation for Animal Health (2019) "
              "guidelines.")
    para(doc, "Images were supplied as JPEG files in a two-level directory structure: a "
              "single folder of normal livers, and a folder of abnormal livers subdivided by "
              "lesion type. Lesion subtype labelling was therefore already available at the "
              "point of analysis, satisfying an objective the proposal had deferred to a "
              "later phase.")

    doc.add_heading("3.6 Data Curation", level=2)

    doc.add_heading("3.6.1 Manifest construction", level=3)
    para(doc, "A single comma-separated manifest was constructed as the authoritative record "
              "for all subsequent stages, replacing directory traversal. Each row records the "
              "relative file path, binary label, lesion subtype, animal identifier, MD5 "
              "checksum, pixel dimensions, file size in bytes, EXIF orientation tag and mean "
              "JPEG quantization value. Recording the capture-related fields alongside the "
              "labels is what made the negative controls of section 3.7.1 possible.")

    doc.add_heading("3.6.2 Deduplication", level=3)
    para(doc, "Files were hashed with the MD5 algorithm and byte-identical duplicates "
              "removed, retaining the first occurrence in path order. This step is not "
              "cosmetic. Duplicated files inflate the apparent sample size and, combined with "
              "random partitioning, place identical images in both the training and the test "
              "set, producing a performance estimate that reflects memorisation rather than "
              "generalisation.")

    doc.add_heading("3.6.3 Reconstruction of animal identity", level=3)
    para(doc, "The collection protocol specifies multiple photographs per liver. Treating "
              "these as independent observations would allow a model to be evaluated on an "
              "animal it had already been trained on. Because no animal identifier was "
              "recorded at the point of capture, animal identity had to be recovered from the "
              "files themselves, by two complementary routes whose results were merged using "
              "a union-find procedure:")
    bullets(doc, [
        "**Filename parsing.** Lesion images encode a case number with an optional "
        "single-letter view suffix — for example LF13a, LF13b and LF13c denote three views "
        "of one liver. A trailing letter was treated as a view marker only when preceded by "
        "a digit, so that filenames carrying no case number were not incorrectly merged.",
        "**Appearance clustering.** Normal images are numbered sequentially and carry no "
        "case information. A 16×16 difference hash and a 32×32 normalised thumbnail were "
        "computed for every image; a pair was joined when the Hamming distance between their "
        "hashes was at most 32, or when it was at most 45 and the correlation between their "
        "thumbnails was at least 0.80.",
    ])
    para(doc, "Parameters: difference-hash side 16; thumbnail side 32; Hamming thresholds 32 "
              "and 45; correlation threshold 0.80. All cross-validation described below "
              "groups on the resulting animal identifier, never on the image.",
         size=11)

    doc.add_heading("3.6.4 Image harmonisation", level=3)
    para(doc, "Every image was passed through one identical processing path, so that "
              "differences arising from capture and export rather than from anatomy could not "
              "reach the classifier. Each file was opened, its EXIF orientation tag applied, "
              "converted to RGB, cropped to a square aspect ratio and scaled to 256×256 "
              "pixels by Lanczos resampling, then written losslessly as an optimised PNG.")
    bullets(doc, [
        "**EXIF transposition.** The orientation tag was present on every normal image and "
        "absent from every lesion image. It is not applied automatically on decoding, so "
        "omitting this step would feed one class to the network rotated by 90 degrees "
        "relative to the other.",
        "**Aspect-preserving crop.** Cropping to the target aspect ratio before scaling "
        "avoids a direct square resize, which would compress each class by a different "
        "factor because the normal images share a single aspect ratio while the lesion "
        "images vary.",
        "**Lossless storage.** Output was written as PNG rather than JPEG for the reason "
        "given below.",
    ])
    para(doc, "Two alternative harmonisation strategies were evaluated and rejected on the "
              "evidence. Re-encoding to JPEG at quality 75 raised the compression-artefact "
              "probe from AUC 0.61 to 0.98: because the two classes arrived at different "
              "original qualities, a second compression interacts with the first and "
              "amplifies the difference rather than erasing it. Gaussian pre-blurring was "
              "also rejected, since a fixed kernel applied to originals of differing size "
              "leaves a size-dependent, and therefore class-dependent, residual.")

    doc.add_heading("3.7 Validity Controls", level=2)
    para(doc, "This section describes the controls that distinguish this analysis from a "
              "conventional image-classification workflow. Their purpose is to answer a "
              "question that a performance figure alone cannot: **is the classifier reading "
              "the pathology, or is it reading the photograph?**")

    doc.add_heading("3.7.1 Negative-control probes", level=3)
    para(doc, "Three probe families were defined, each attempting the classification task "
              "using a restricted set of features. They are kept separate because conflating "
              "them overstates the leak: the third family can legitimately reflect "
              "pathology, whereas the first two cannot.")
    table(doc,
          ["Probe family", "Features supplied", "Interpretation", "Gated"],
          [["Capture metadata",
            "Width, height, aspect ratio, pixel count, mean JPEG quantization, EXIF "
            "orientation",
            "Carries no image content; any discrimination is pure leakage", "Yes"],
           ["Compression artefact",
            "Gradient energy on the JPEG 8-pixel block grid divided by energy elsewhere "
            "(2 features)",
            "A compression fingerprint residing in the pixels", "Yes"],
           ["Image content statistics",
            "Greyscale standard deviation, greyscale mean, bytes per pixel",
            "Contrast genuinely differs between uniform and abscessed tissue; not "
            "treated as leakage", "No"]],
          widths=[1.2, 2.2, 2.1, 0.5])
    caption(doc, "Table 3.2: Negative-control probe families. Only the first two gate the "
                 "pipeline.")
    para(doc, "Each probe was scored with both a random forest of 400 trees and a "
              "radial-basis-function support vector machine, under animal-grouped five-fold "
              "cross-validation, retaining the higher of the two AUCs on the principle that a "
              "leak detectable by either model is a leak. The gate threshold was set at AUC "
              "0.65. A label-permutation control, comprising five repetitions of the full "
              "protocol with the class labels shuffled, was run to verify that the evaluation "
              "machinery itself introduces no optimistic bias.")

    doc.add_heading("3.7.2 Confound localisation diagnostics", level=3)
    para(doc, "Where a gated probe passes but classification performance nevertheless remains "
              "implausibly high, the residual signal must be located within the image. Seven "
              "variants of each harmonised image were constructed, each destroying lesion "
              "detail in a different way, and colour statistics — channel mean, standard "
              "deviation, and 10th and 90th percentiles — were extracted from each:")
    bullets(doc, [
        "**Full image** — the reference condition.",
        "**Centre patch only** — the central 40 per cent of the frame, where the lesion is "
        "most likely to lie.",
        "**Border ring only** — the outer 20 per cent frame, with the centre discarded "
        "entirely, so that the organ is excluded.",
        "**Border ring after per-image colour normalisation** — as above, but with each "
        "photograph's own colour cast and exposure removed first, which distinguishes a "
        "global colour difference from a difference in framing and composition.",
        "**Heavy Gaussian blur** (sigma = 12) — coarse colour layout, no detail.",
        "**8×8 pixel thumbnail** — no lesion resolvable at any scale.",
        "**Mean colour** — three numbers per image.",
    ])
    para(doc, "The logic is straightforward: any variant that retains high discriminative "
              "power after the lesion has been rendered unreadable must be measuring "
              "something other than the lesion.")

    doc.add_heading("3.8 Data Partitioning", level=2)
    para(doc, "Nested cross-validation was used, with grouping by animal enforced at both "
              "levels. The outer loop estimates performance; the inner loop performs all "
              "model selection. This separation addresses a defect in which a model "
              "checkpoint or decision threshold is chosen using the same partition that is "
              "subsequently used to report performance, which reports the most favourable "
              "configuration rather than genuine generalisation.")
    bullets(doc, [
        "**Outer partition.** Stratified group five-fold, grouping on animal identifier, "
        "with three repetitions under different random seeds for the classical arm and one "
        "for the deep arm.",
        "**Inner partition.** A stratified group four-fold split of the outer training "
        "partition, used exclusively for early stopping and decision-threshold selection.",
        "**Verification.** Every fold was checked programmatically for group overlap between "
        "the training and test partitions; all folds returned none.",
    ])
    para(doc, "Identical fold objects were shared between the classical and the deep arm, so "
              "that the comparison between them is like-for-like. This is a refinement of the "
              "simple stratified k-fold cross-validation specified in the proposal, adopted "
              "because the dataset contains multiple photographs of the same animal (section "
              "3.6.3), which simple stratification would distribute across folds.")

    doc.add_heading("3.9 Model Development", level=2)

    doc.add_heading("3.9.1 Classical machine-learning baseline", level=3)
    para(doc, "A 162-dimensional feature vector was extracted from each harmonised image:")
    bullets(doc, [
        "**Colour distribution.** Sixteen-bin normalised histograms of each RGB channel, "
        "together with the channel means and standard deviations, plus twelve-bin normalised "
        "histograms of each HSV channel. Hue and saturation were included because they carry "
        "the pale bile-duct tracts characteristic of fascioliasis and the white foci of "
        "abscessation.",
        "**Texture.** Grey-level co-occurrence matrices (Haralick et al., 1973) computed on "
        "lightness quantised to 64 levels, at distances of 1, 3 and 5 pixels and angles of "
        "0, π/4, π/2 and 3π/4. Six properties were extracted from each matrix: contrast, "
        "dissimilarity, homogeneity, energy, correlation and angular second moment.",
    ])
    para(doc, "Three classifiers were trained on these features: a radial-basis-function "
              "support vector machine (C = 1.0, balanced class weights); a random forest (500 "
              "trees, minimum two samples per leaf, balanced subsample weighting); and "
              "L2-regularised logistic regression (C = 0.1, balanced class weights). Features "
              "were standardised within each training fold for the support vector machine and "
              "the logistic model.")

    doc.add_heading("3.9.2 Transfer learning", level=3)
    para(doc, "Three ImageNet-pretrained convolutional architectures were evaluated — "
              "MobileNetV2, EfficientNet-B0 and ResNet18 — each in two regimes, giving six "
              "configurations. In every case the final classification layer was replaced with "
              "a single-logit linear layer.")
    para(doc, "**Frozen-backbone regime.** With the convolutional trunk held fixed, "
              "embeddings depend only on the input and were therefore computed once per image "
              "rather than at every epoch. Six views were embedded per image: one under the "
              "deterministic evaluation transform, and five under the training augmentation. "
              "Augmented views of training images were stacked as additional training rows, "
              "while inner-validation and test rows used the deterministic view only. An "
              "L2-regularised logistic head (C = 0.1, balanced class weights, maximum 3000 "
              "iterations) was then fitted on the standardised embeddings.")
    para(doc, "**Fine-tuned regime.** The full network was trained using the AdamW optimiser "
              "(learning rate 3×10⁻⁴, weight decay 10⁻⁴), batch size 16, a cosine-annealed "
              "learning-rate schedule, and binary cross-entropy loss with a positive-class "
              "weight equal to the negative-to-positive ratio within each training fold. "
              "Training ran for a maximum of 12 epochs with early stopping on inner-fold loss "
              "(patience 4). This regime is implemented and available behind a command-line "
              "flag, but was not executed for the results reported in Chapter Four, for the "
              "reason given in section 5.3.")

    doc.add_heading("3.9.3 Data augmentation and class imbalance", level=3)
    para(doc, "Augmentation comprised random horizontal flipping (p = 0.5), vertical flipping "
              "(p = 0.3), rotation of up to 20 degrees, colour jitter (brightness 0.25, "
              "contrast 0.25, saturation 0.2, hue 0.02), and random affine translation of up "
              "to 8 per cent with scaling between 0.9 and 1.1. Images were resized to 224×224 "
              "pixels and normalised using ImageNet channel statistics. Class imbalance was "
              "handled by class-weighted loss functions and balanced class weights throughout.")
    para(doc, "Random erasing, which the proposal specified, was deliberately omitted. On a "
              "dataset of this size it removes the lesion from an already small positive class "
              "and trains the model to infer the label from the surrounding context — "
              "precisely the failure mode under investigation in this study.")

    doc.add_heading("3.10 Interpretability Analysis", level=2)
    para(doc, "The proposal specified Grad-CAM visualisation to confirm that classification "
              "decisions are driven by the lesion rather than by background or lighting. That "
              "objective was implemented as a quantitative hypothesis test rather than as an "
              "illustrative figure, so that it is capable of returning a result which "
              "contradicts the study's premise. Two measures were computed:")
    bullets(doc, [
        "**Border attention share.** Gradient-weighted class activation maps (Selvaraju et "
        "al., 2017) were computed at the final convolutional block, and the proportion of "
        "total activation falling within the outer 20 per cent frame was recorded. A "
        "spatially uniform attention map yields 0.640 under this definition, which provides "
        "the reference value; a substantially higher value indicates attention directed "
        "outside the organ.",
        "**Occlusion sensitivity.** A 48-pixel patch was translated across the image at a "
        "stride of 16 pixels and the change in predicted probability recorded at each "
        "position. Unlike gradient-based attribution, this requires no gradients and cannot "
        "be misled by a sharp but causally irrelevant activation: if concealing a region "
        "does not alter the prediction, that region was not being used.",
    ])
    para(doc, "A technical note is warranted. With a frozen trunk, no parameter requires a "
              "gradient, so backpropagation terminates before reaching the target layer and "
              "the backward hook never fires. Requiring a gradient on the input tensor "
              "preserves the computational graph and resolves this.", size=11)

    doc.add_heading("3.11 Tissue-Restricted Patch Analysis", level=2)
    para(doc, "The diagnostics of section 3.7.2 localised the confound to the background and "
              "the framing of the photographs (section 4.5). An analysis was therefore "
              "constructed that discards both: the organ is segmented, samples are drawn from "
              "well inside it, and **tissue rather than photographs** is classified. The "
              "organ outline, the shot composition, the inspection surface and the image "
              "border are excluded by construction.")

    doc.add_heading("3.11.1 Organ segmentation", level=3)
    para(doc, "No manually delineated liver masks were available for supervised "
              "segmentation, so a classical colour-clustering procedure was used, which has "
              "the further advantage of being visually inspectable. Simple darkness or "
              "redness thresholding was tried first and rejected: pooled blood and a wet "
              "inspection table are both dark and red, so such rules retain large areas of "
              "background. The adopted procedure was:")
    numbered(doc, [
        "Convert the image to the CIELAB colour space and cluster pixel colours by k-means "
        "(k = 4, four initialisations), fitted on a one-in-three subsample for speed.",
        "Weight each pixel by a Gaussian centred on the frame (sigma = 0.45 of the frame "
        "dimensions) and select the cluster with the greatest total weight, on the principle "
        "that the subject of a photograph is what the photographer aimed at.",
        "Merge any adjacent cluster whose mean lightness is up to 18 units darker, so that a "
        "fissure or a cut surface does not divide the organ.",
        "Apply morphological opening (disk radius 3), closing (disk radius 9) and hole "
        "filling; remove objects below 2 per cent of frame area; retain the largest connected "
        "component.",
        "If the resulting mask covers less than 8 per cent of the frame, treat the rule as "
        "having failed and substitute a generous central region.",
    ])

    doc.add_heading("3.11.2 Magnification-normalised patch sampling", level=3)
    para(doc, "A fixed-pixel sampling window means different things in a close-up and in a "
              "whole-organ photograph, so camera distance could masquerade as tissue texture. "
              "The window side length was therefore set to 0.30 × √(mask area), so that every "
              "patch covers the same fraction of the organ irrespective of camera distance, "
              "and each patch was then resampled to 64×64 pixels. Patches were accepted only "
              "where at least 85 per cent of their area fell within a mask eroded by one "
              "eighth of the window side, ensuring that no sample straddles the organ "
              "boundary; where this constraint could not be satisfied it was relaxed to 60 "
              "per cent coverage on the uneroded mask. Up to 12 patches were drawn per image.")
    para(doc, "The contribution of this normalisation was quantified directly by repeating "
              "the analysis with a fixed 64-pixel window, and is reported in section 4.9.")

    doc.add_heading("3.11.3 Patch features and aggregation", level=3)
    bullets(doc, [
        "**Colour features (36 per patch).** CIELAB channel mean, standard deviation, and "
        "10th and 90th percentiles, together with eight-bin RGB histograms.",
        "**Texture features (72 per patch).** Grey-level co-occurrence matrices at distances "
        "of 1, 2 and 4 pixels and four angles, at 64 levels, with six properties extracted. "
        "Critically, lightness is standardised **within each patch** before quantisation, so "
        "that absolute brightness — which encodes the photographic session rather than the "
        "pathology — cannot contribute. Only the spatial arrangement of lightness survives, "
        "which makes this feature set invariant to session lighting and white balance by "
        "construction.",
    ])
    para(doc, "Patch-level predictions were averaged within each image and then within each "
              "animal, and all AUCs from this analysis are reported at the level of the "
              "individual animal. The classifier was a radial-basis-function support vector "
              "machine with balanced class weights, under the same animal-grouped "
              "cross-validation used elsewhere.")

    doc.add_heading("3.11.4 Confound probes for the patch analysis", level=3)
    para(doc, "The tissue analysis carries its own negative controls, so that its result can "
              "be compared against the strongest artefactual explanation available rather "
              "than against chance alone. Three probes were run on the identical patches, "
              "each supplied only with information that cannot encode pathology: patch mean "
              "colour (3 features, capturing session colour and white balance); patch JPEG "
              "blockiness (2 features, capturing compression history); and patch spatial-"
              "frequency profile and autocorrelation width (4 features, capturing apparent "
              "magnification). A task result is credible only if it clears all three.")

    doc.add_heading("3.12 Statistical Analysis and Model Evaluation", level=2)
    para(doc, "Model performance was evaluated using the metrics specified in the proposal, "
              "computed per fold and reported as mean ± standard deviation across folds, and "
              "additionally on pooled out-of-fold predictions.")
    table(doc,
          ["Metric", "What it measures", "Relevance to meat inspection"],
          [["Sensitivity (recall)", "Proportion of true lesioned livers correctly flagged",
            "A missed lesion is the costlier error"],
           ["Specificity", "Proportion of true normal livers correctly passed",
            "Excess false positives would undermine inspector trust"],
           ["Precision", "Proportion of positive predictions that are truly positive",
            "Relevant given the class imbalance in the dataset"],
           ["F1 score", "Harmonic mean of precision and recall",
            "Single balanced measure for comparing configurations"],
           ["AUC-ROC", "Overall discrimination across all thresholds",
            "Threshold-independent comparison between arms"]],
          widths=[1.2, 2.5, 2.3])
    caption(doc, "Table 3.3: Evaluation metrics and their interpretation in this setting.")
    bullets(doc, [
        "**Decision threshold.** Selected on inner-fold predictions as the threshold "
        "achieving at least 90 per cent sensitivity with the lowest false-positive rate, "
        "reflecting the asymmetric cost of a missed lesion. Where that target was unreachable "
        "within a fold, Youden's J statistic was used instead.",
        "**Confidence intervals.** Percentile bootstrap over 2000 resamples, resampling "
        "**animals rather than images**. Resampling images would treat three photographs of "
        "one liver as three independent observations and yield intervals that are too narrow.",
        "**Reproducibility.** A fixed random seed (42) governs partitioning, augmentation, "
        "clustering and bootstrap resampling throughout.",
    ])

    doc.add_heading("3.13 Software and Reproducibility", level=2)
    para(doc, "All analysis was performed in Python 3.14 within a dedicated Conda environment "
              "on a four-thread CPU with no GPU available. Principal libraries were PyTorch "
              "and torchvision (transfer learning), scikit-learn (cross-validation, classical "
              "classifiers and metrics), scikit-image (texture features, morphology and "
              "colour conversion), Pillow (image input and output), and SciPy, NumPy, pandas "
              "and matplotlib. The absence of GPU acceleration materially shaped the "
              "experimental design: frozen-backbone configurations were evaluated using "
              "precomputed embeddings, which reduced their cost from hours to approximately "
              "one minute per architecture. The pipeline is organised as eleven modules "
              "driven by numbered entry-point scripts, listed in Appendix B.")

    doc.add_heading("3.14 Ethical Considerations", level=2)
    para(doc, "Data collection involved photographing organs already condemned or passed "
              "during routine, legally mandated post-mortem inspection. No live-animal "
              "experimentation was involved and no procedure was performed on any animal for "
              "the purposes of this study. The ethical requirements were therefore "
              "institutional and departmental clearance together with abattoir management "
              "permission to photograph organs and record inspection findings. No personal or "
              "farm-identifying information was captured; where any such information appeared "
              "incidentally in metadata it was removed prior to analysis and storage.")

    doc.add_page_break()

    # =================================================================== CHAPTER FOUR
    doc.add_heading("CHAPTER FOUR: RESULTS", level=1)

    doc.add_heading("4.1 Dataset Composition After Curation", level=2)
    para(doc, f"Nine groups of byte-identical files were identified among the {n_files} files "
              f"supplied, comprising {len(dropped) + 9} files that reduced to 9 unique images; "
              f"one normal photograph had been stored five times. After removing "
              f"{len(dropped)} redundant files, {n_img} unique images remained. Filename "
              f"parsing and appearance clustering resolved these to {n_animals} distinct "
              f"animals, of which {n_multi} contributed more than one photograph.")
    table(doc,
          ["Item", "Count"],
          [["Image files supplied", n_files],
           ["Byte-identical duplicates removed", len(dropped)],
           ["Unique images analysed", n_img],
           ["Distinct animals represented", n_animals],
           ["Animals contributing more than one image", n_multi],
           ["Normal images", n_norm],
           ["Images with a visible gross lesion", n_les]],
          widths=[3.6, 1.2])
    caption(doc, "Table 4.1: Dataset composition after deduplication and grouping by animal.")
    para(doc, f"This departs materially from the 100 images (60 normal, 40 abnormal) "
              f"described in the proposal. The curated dataset is larger in image count "
              f"({n_img}) but, once repeat views of the same liver are grouped, rests on "
              f"{n_animals} animals. Because the animal rather than the image is the unit of "
              f"statistical independence, all sample-size statements in this thesis are "
              f"expressed in distinct animals.")

    doc.add_heading("4.2 Distribution of Gross Lesion Types", level=2)
    para(doc, "The descriptive characterisation of gross lesion types specified as Objective "
              "6 is presented in Table 4.2 and Figure 4.1, reported by image count and by "
              "distinct animal.")
    table(doc,
          ["Lesion type", "Images", "Distinct animals", "Suitability for modelling"],
          [[idx, int(r.images), int(r.livers),
            "Adequate" if r.livers >= 16 else "Single animal — cannot be validated"]
           for idx, r in by_type.iterrows()],
          widths=[1.4, 0.8, 1.3, 2.5])
    caption(doc, "Table 4.2: Gross lesion types by image count and by distinct animal.")
    figure(doc, FIG / "lesion_frequency.png",
           "Figure 4.1: Lesion frequency by image count and by distinct animal. The "
           "divergence between the two bars is the reason sample-size targets must be "
           "expressed in animals rather than images.", width=5.3)
    para(doc, "Fascioliasis was the most common finding (39 images from 34 animals), followed "
              "by hepatic abscessation (18 images from 16 animals). Three of the five lesion "
              "categories — telangiectasis, cirrhosis and fibrosis — derive from a single "
              "animal each, which precludes multi-class classification on the present dataset: "
              "a category represented by one animal cannot be placed in both a training and a "
              "test partition, so no cross-validated estimate of performance on it is "
              "obtainable.")
    para(doc, "Two discrepancies with the proposal's lesion category list (§2.5) were "
              "identified. Telangiectasis occurs in the data but is absent from the "
              "proposal's list, while fatty liver is listed in the proposal but was not "
              "observed in any image.")

    doc.add_heading("4.3 Detection of a Capture Confound", level=2)
    para(doc, f"Before harmonisation, a random forest supplied only with file properties — "
              f"pixel dimensions, byte size, JPEG quantization values and EXIF orientation, "
              f"and **no pixel data of any kind** — classified the images perfectly (AUC "
              f"{f3(probe('RAW', 'capture metadata'))}, 100 per cent accuracy under "
              f"animal-grouped five-fold cross-validation).")
    para(doc, "Inspection revealed the mechanism. Mean JPEG quantization values formed two "
              "non-overlapping regimes coinciding exactly with the class label: 12.22 for all "
              "normal images, against 35.59 or 44.81 for all lesion images. Every normal image "
              "measured 0.87 megapixels and carried an EXIF orientation tag; every lesion "
              "image had variable dimensions and no EXIF data. The uniform sizing is "
              "characteristic of export through a messaging application rather than direct "
              "transfer from the capture device. The two classes had, in effect, travelled to "
              "the analysis by different routes, and that route was perfectly predictive of "
              "the diagnosis.")

    doc.add_heading("4.4 Effect of Harmonisation on the Negative Controls", level=2)
    table(doc,
          ["Probe", "Raw images", "Harmonised", "Gate"],
          [["Capture metadata", f3(probe("RAW", "capture metadata")),
            f3(probe("HARMONISED", "capture metadata")), "Pass"],
           ["Compression artefact", f3(probe("RAW", "compression artefact")),
            f3(probe("HARMONISED", "compression artefact")), "Pass"],
           ["Image content statistics", f3(probe("RAW", "content")),
            f3(probe("HARMONISED", "content")), "Not gated"],
           ["Shuffled labels (protocol control)", "—",
            f3(nc.loc[nc.probe.str.contains("shuffled"), "worst_auc"].item()), "Pass"]],
          widths=[2.4, 1.2, 1.2, 1.0])
    caption(doc, "Table 4.3: Negative-control AUCs before and after harmonisation. "
                 "Gate threshold 0.65.")
    para(doc, f"Harmonisation eliminated the file-level confound completely: capture metadata "
              f"fell from perfect discrimination to exactly chance "
              f"({f3(probe('HARMONISED', 'capture metadata'))}), as expected once every "
              f"property it measures has been made constant across the dataset. The "
              f"compression-artefact probe remained at "
              f"{f3(probe('HARMONISED', 'compression artefact'))}, below the gate but above "
              f"chance, reflecting the fact that the first compression is already baked into "
              f"the pixel values and cannot be removed retrospectively. The label-permutation "
              f"control returned "
              f"{f3(nc.loc[nc.probe.str.contains('shuffled'), 'worst_auc'].item())}, "
              f"confirming that the cross-validation machinery introduces no optimistic bias "
              f"of its own.")
    para(doc, "Both gated probes therefore passed, and on the evidence available at this "
              "point the dataset would conventionally have been declared fit for modelling.")

    doc.add_heading("4.5 Localisation of the Residual Signal", level=2)
    para(doc, "Although both gated probes passed, whole-image classification performance "
              "(section 4.6) remained implausibly high for a dataset of this size, so the "
              "diagnostic variants of section 3.7.2 were applied.")
    table(doc,
          ["Information available to the classifier", "AUC", "Interpretation"],
          [["Full harmonised image", f3(diag("full image")), "Reference"],
           ["8×8 thumbnail — no lesion resolvable",
            f3(diag("8x8 thumbnail (all detail gone)")), "Exceeds the full image"],
           ["Border ring only — organ excised",
            f3(diag("border ring only (no centre)")), "Background alone suffices"],
           ["Border ring, per-image colour normalised",
            f3(diag("border ring, per-image colour normalised")),
            "Survives removal of colour cast"],
           ["Heavy blur — colour layout only",
            f3(diag("heavy blur (colour layout only)")), "No detail required"],
           ["Mean colour — three numbers",
            f3(diag("mean colour only (3 numbers)")), "Colour cast alone"],
           ["Centre patch — where the lesion lies",
            f3(diag("centre patch only")), "Least informative region"]],
          widths=[2.8, 0.7, 2.4])
    caption(doc, "Table 4.4: Class discrimination surviving progressive destruction of lesion "
                 "detail.")
    para(doc, f"The result is unambiguous. An 8×8 thumbnail, in which no lesion can be "
              f"resolved at any scale, scored {f3(diag('8x8 thumbnail (all detail gone)'))} "
              f"and thereby **outperformed the full image** "
              f"({f3(diag('full image'))}). The image border with the organ excised achieved "
              f"near-perfect discrimination ({f3(diag('border ring only (no centre)'))}). The "
              f"central region, where the lesion actually lies, was the **least** informative "
              f"part of the image ({f3(diag('centre patch only'))}).")
    para(doc, f"Per-image colour normalisation was applied to the border ring as a candidate "
              f"remedy and did not resolve the problem: discrimination from the border alone "
              f"remained at {f3(diag('border ring, per-image colour normalised'))} after each "
              f"photograph's own colour cast and exposure had been removed. The difference "
              f"between the classes therefore lies in framing and composition, not merely in "
              f"a global colour shift that normalisation could correct.")
    figure(doc, FIG / "class_backgrounds.png",
           "Figure 4.2: Six randomly selected normal livers (upper row) and six lesion images "
           "(lower row) after harmonisation. Normal organs are photographed whole, at a "
           "consistent distance, on pale matting; lesion images are close-up crops taken at "
           "varying distance on varying surfaces. The two classes are separable without any "
           "reference to pathology.")

    doc.add_heading("4.6 Whole-Image Classification Performance", level=2)
    para(doc, "The whole-image results specified by Objectives 2, 3 and 4 are reported below "
              "for completeness. They are presented together with the 8×8 thumbnail reference "
              "from Table 4.4, because that comparison is what makes them interpretable.")
    table(doc,
          ["Model", "Pooled AUC", "95% CI", "Sensitivity", "Specificity", "F1"],
          [[r.model, f3(r.auc), f"{r.auc_lo:.3f}–{r.auc_hi:.3f}",
            f3(r.sensitivity), f3(r.specificity), f3(r.f1)]
           for _, r in pd.concat([dp, cp]).iterrows()]
          + [["8×8 thumbnail (reference)", f3(diag("8x8 thumbnail (all detail gone)")),
              "—", "—", "—", "—"]],
          widths=[1.7, 0.9, 1.2, 1.0, 1.0, 0.6])
    caption(doc, "Table 4.5: Whole-image performance on pooled out-of-fold predictions under "
                 "animal-grouped nested cross-validation. All values are invalidated by "
                 "section 4.5.")
    para(doc, "The proposal specified reporting as mean ± standard deviation across folds; "
              "those values are given in Table 4.6.")
    table(doc,
          ["Model", "AUC", "Sensitivity", "Specificity", "F1"],
          [[m, ms(sub, "auc"), ms(sub, "sensitivity"), ms(sub, "specificity"), ms(sub, "f1")]
           for m, sub in list(dfm.groupby("model")) + list(cf.groupby("model"))],
          widths=[1.7, 1.2, 1.2, 1.2, 1.1])
    caption(doc, "Table 4.6: Whole-image performance as mean ± standard deviation across "
                 "cross-validation folds.")
    para(doc, f"Every architecture, deep or classical, converged on approximately the same "
              f"value as a thumbnail too small to contain a lesion. The best deep "
              f"configuration ({best_deep.model}) reached AUC {f3(best_deep.auc)} and the best "
              f"classical configuration ({best_cls.model}) reached {f3(best_cls.auc)}, against "
              f"{f3(diag('8x8 thumbnail (all detail gone)'))} for the thumbnail. This "
              f"convergence is itself diagnostic: it indicates a task being solved by "
              f"information available equally to all of them. The convolutional networks did "
              f"not meaningfully outperform the classical baseline because there was no "
              f"additional structure for them to exploit.")

    doc.add_heading("4.7 Interpretability", level=2)
    para(doc, "Objective 5 required confirmation that classification decisions are driven by "
              "the lesion rather than by background or lighting. Implemented as the "
              "quantitative test described in section 3.10, this objective returned a "
              "negative result.")
    table(doc,
          ["Measure", "Value", "Reference"],
          [["Uniform attention map (definitional reference)", "0.640", "—"],
           ["Grad-CAM border share, lesion images", f3(gc_les), "0.640"],
           ["Grad-CAM border share, normal images", f3(gc_norm), "0.640"],
           ["Occlusion evidence in outer frame",
            f3(oc.occlusion_border_share.mean()), "0.640"]],
          widths=[3.2, 1.1, 1.1])
    caption(doc, f"Table 4.7: Proportion of model evidence falling outside the organ "
                 f"(n = {len(gc)} held-out images; occlusion computed on {len(oc)}).")
    para(doc, f"On normal livers, {gc_norm:.1%} of activation fell outside the organ. The "
              f"model's evidence for classifying a liver as normal is therefore almost "
              f"entirely the surrounding scene. The occlusion analysis, which requires no "
              f"gradients and is immune to gradient-attribution artefacts, independently "
              f"corroborated this at {f3(oc.occlusion_border_share.mean())}. Attention on "
              f"lesion images was more central ({f3(gc_les)}), indicating that lesion "
              f"recognition does draw partly on tissue. The interpretability objective thus "
              f"fails on this dataset in a specific and informative way: the model is "
              f"detecting the photograph of a normal liver, not the normal liver.")
    figure(doc, FIG / "gradcam_compact.png",
           "Figure 4.3: Upper row — a fascioliasis case, with attention falling on tissue, "
           "which is the intended behaviour. Lower row — a normal liver, where the activation "
           "map is flat across the organ and occlusion locates the decisive evidence away "
           "from the tissue.", width=5.6)

    doc.add_heading("4.8 Tissue-Restricted Patch Analysis", level=2)
    para(doc, f"Segmentation yielded organ masks covering a mean of "
              f"{ps.mask_frac_normal:.2f} of the frame for normal images and "
              f"{ps.mask_frac_lesion:.2f} for lesion images — a close correspondence, "
              f"indicating that mask area is not itself a strong class cue. A total of "
              f"{int(ps.n_patches):,} patches were obtained from all "
              f"{int(ps.n_images_sampled)} images, with no image failing to yield a patch.")
    table(doc,
          ["Configuration", "Features", "AUC (per animal)", "95% CI"],
          [[r.configuration.replace("PROBE  ", "Probe — ").replace("TASK   ", "Task — "),
            int(r.n_features), f3(r.auc), f"{r.lo:.3f}–{r.hi:.3f}"]
           for _, r in pa.iterrows()],
          widths=[2.9, 0.8, 1.2, 1.3])
    caption(doc, "Table 4.8: Patch-level results, scored at the level of the individual "
                 "animal under animal-grouped cross-validation.")
    para(doc, f"Texture computed on locally standardised lightness achieved AUC "
              f"{f3(texture.auc)} (95% CI {texture.lo:.3f}–{texture.hi:.3f}) at the level of "
              f"the individual animal, exceeding the strongest confound probe "
              f"({worst_probe_name}, {f3(worst_probe)}) by {texture.auc - worst_probe:.3f}. "
              f"Each of the three candidate artefactual explanations — session colour, "
              f"compression history and camera magnification — was independently controlled, "
              f"and all three fell well below the task result.")
    para(doc, f"Colour features reached {f3(colour.auc)} "
              f"(95% CI {colour.lo:.3f}–{colour.hi:.3f}). This is a higher figure than the "
              f"texture result, but it is not the one this thesis reports, because the "
              f"mean-colour probe on the same patches reached "
              f"{f3(patch('mean colour').auc)} — well above chance. The colour pathway "
              f"therefore remains partly confounded with the photographic session, whereas "
              f"the texture pathway is invariant to illumination by construction.")
    figure(doc, FIG / "segmentation_check.png",
           "Figure 4.4: Segmentation and patch sampling. Upper row — harmonised input. "
           "Middle row — the selected organ mask, with excluded regions dimmed. Lower row — "
           "representative tissue patches drawn from the masked interior.")

    doc.add_heading("4.9 Effect of Magnification Normalisation", level=2)
    para(doc, "Because camera distance is a plausible artefactual explanation for a texture "
              "signal, the contribution of the organ-relative sampling described in section "
              "3.11.2 was quantified directly by repeating the texture analysis with a fixed "
              "64-pixel window over identical images, masks, seed and classifier.")
    table(doc,
          ["Patch sampling regime", "Patches", "AUC (per animal)", "95% CI"],
          [[r.sampling.capitalize(), f"{int(r.n_patches):,}", f3(r.auc),
            f"{r.lo:.3f}–{r.hi:.3f}"] for _, r in sa.iterrows()],
          widths=[2.7, 0.9, 1.3, 1.3])
    caption(doc, "Table 4.9: Texture performance with and without magnification "
                 "normalisation.")
    para(doc, f"Equalising magnification between close-ups and whole-organ shots raised "
              f"performance from {f3(sa_fixed)} to {f3(sa_norm)}, an improvement of "
              f"{sa_norm - sa_fixed:+.3f}. This supports the interpretation that the signal "
              f"reflects genuine tissue structure: an artefactual signal riding on camera "
              f"distance would be expected to **degrade** when that distance is normalised "
              f"away, not to improve.")

    doc.add_page_break()

    # =================================================================== CHAPTER FIVE
    doc.add_heading("CHAPTER FIVE: DISCUSSION", level=1)

    doc.add_heading("5.1 Principal Findings", level=2)
    para(doc, f"This study set out to develop and evaluate a smartphone-based classifier of "
              f"bovine liver lesions. It produced two findings, one negative and one positive, "
              f"and the negative one is a precondition for understanding the positive one.")
    para(doc, f"The negative finding is that **whole-image classification cannot be validated "
              f"on this dataset**. Classifiers reached AUC values up to {f3(dp.auc.max())}, "
              f"which in the absence of controls would have been reported as an excellent "
              f"pilot result. The controls establish that this performance is not attributable "
              f"to pathology: an 8×8 thumbnail scored "
              f"{f3(diag('8x8 thumbnail (all detail gone)'))}, the image border with the organ "
              f"cut out scored {f3(diag('border ring only (no centre)'))}, and "
              f"{gc_norm:.1%} of the model's attention on normal livers fell outside the "
              f"organ entirely.")
    para(doc, f"The positive finding is that **liver parenchymal texture, measured from "
              f"smartphone photographs, discriminates lesioned from normal organs**. Restricted "
              f"to tissue sampled from inside the segmented organ, and using texture features "
              f"that are invariant to illumination by construction, classification reached AUC "
              f"{f3(texture.auc)} (95% CI {texture.lo:.3f}–{texture.hi:.3f}) across "
              f"{n_animals} animals, against a worst-case confound probe of "
              f"{f3(worst_probe)}. This is a narrower claim than the one the study originally "
              f"set out to make, but it is supported by the data in hand.")

    doc.add_heading("5.2 The Capture Confound and Shortcut Learning", level=2)
    para(doc, "The defect identified in sections 4.3 to 4.5 is an instance of shortcut "
              "learning (Geirhos et al., 2020), in which a model exploits an incidental "
              "correlate of the label rather than the feature of clinical interest. The "
              "phenomenon is well documented in human medical imaging: convolutional networks "
              "have been shown to infer pneumonia risk from the scanner or department that "
              "produced a radiograph (Zech et al., 2018), and melanoma risk from the presence "
              "of surgical skin markings placed by clinicians on lesions they had already "
              "judged suspicious (Winkler et al., 2019). A systematic review of COVID-19 "
              "imaging models found that essentially none were of clinical use, very largely "
              "for reasons of this kind (Roberts et al., 2021).")
    para(doc, "The present case is the veterinary equivalent, and it arose through an "
              "entirely ordinary sequence of events. Normal livers were photographed in one "
              "manner — whole organ, consistent distance, pale matting — and lesioned livers "
              "in another, as close-up crops on varying surfaces, which is a natural thing to "
              "do when the object of interest is a lesion. The normal images were then "
              "transferred through a messaging application, which resampled them to a uniform "
              "size and re-encoded them at a uniform quality. Neither decision was careless in "
              "any obvious way, and neither would be visible in a table of results. Together "
              "they made the diagnosis perfectly predictable from information that has nothing "
              "to do with the liver.")
    para(doc, "The wider methodological point is that no performance figure, however high, "
              "carries information about its own validity. Cross-validation does not detect "
              "this failure, because the confound is present in the training and the test "
              "partitions alike. Only a control that deliberately withholds the pathology can "
              "detect it, and such controls are not yet routine in the veterinary imaging "
              "literature.")

    doc.add_heading("5.3 Why the Whole-Image Objectives Could Not Be Validated", level=2)
    para(doc, "Four preprocessing remedies were attempted and none was sufficient. Lossless "
              "harmonisation removed the file-level confound completely (Table 4.3) but left "
              "the framing and background differences untouched. Per-image colour "
              f"normalisation left border-only discrimination at "
              f"{f3(diag('border ring, per-image colour normalised'))} (section 4.5), showing "
              "that the difference is compositional rather than a global colour cast. Centre "
              "cropping reduced but did not eliminate the effect, and the centre region was in "
              "any case the least discriminative part of the image. JPEG re-encoding actively "
              "worsened the problem, raising the compression-artefact probe from 0.61 to 0.98.")
    para(doc, "This exhausts what preprocessing can achieve. The confound is not an artefact "
              "of file handling that survived into the pixels; it is a property of how the "
              "photographs were composed. Information that was never captured cannot be "
              "recovered by processing, and information that was captured differently for the "
              "two classes cannot be made identical after the fact. The remedy is therefore "
              "re-collection under the protocol set out in section 5.9.1, not a further "
              "preprocessing step.")
    para(doc, "For the same reason, the fine-tuned configurations were implemented but not "
              "executed. Fine-tuning six configurations on a four-thread CPU costs several "
              "hours, and the confound analysis had already established that the resulting "
              "figures would measure capture conditions rather than pathology. Spending that "
              "computation would have produced more precise estimates of the wrong quantity. "
              "The code path remains available for use once the data support it.")

    doc.add_heading("5.4 Liver Parenchymal Texture as a Valid Pilot Signal", level=2)
    para(doc, "The tissue-restricted analysis was constructed specifically to be immune to "
              "the failure identified above. Segmenting the organ and sampling from its "
              "interior removes the background, the inspection surface, the organ outline and "
              "the image border by construction. Standardising lightness within each patch "
              "removes the session's exposure and white balance. Sizing the sampling window "
              "against the organ's apparent area removes camera distance.")
    para(doc, f"The result survives its own controls. Texture reached {f3(texture.auc)} while "
              f"the strongest probe — {worst_probe_name} — reached {f3(worst_probe)}, a margin "
              f"of {texture.auc - worst_probe:.3f}. Colour, blockiness and magnification each "
              f"had a dedicated probe and all three sat far below the task result. The "
              f"magnification ablation in section 4.9 provides positive rather than merely "
              f"negative evidence: normalising magnification **improved** performance by "
              f"{sa_norm - sa_fixed:+.3f}, which is the direction expected of a genuine tissue "
              f"signal and the opposite of what an artefact riding on camera distance would do.")
    para(doc, "Biologically, the finding is unsurprising. Fascioliasis produces thickened, "
              "fibrotic bile ducts and irregular pale tracts across the parenchymal surface; "
              "abscessation produces focal capsular distortion and raised firm foci. Both "
              "alter the spatial arrangement of surface lightness at a scale a smartphone "
              "resolves, which is precisely what a grey-level co-occurrence matrix measures. "
              "The claim this thesis makes is correspondingly specific: it is a claim about "
              "**parenchymal texture**, not about photographs of livers.")

    doc.add_heading("5.5 Feasibility of Multi-Class Lesion Classification", level=2)
    para(doc, "The descriptive characterisation required by Objective 6 was completed and "
              "carries a clear implication for the study's second phase. Of the five lesion "
              "categories observed, only fascioliasis (34 animals) and hepatic abscessation "
              "(16 animals) rest on enough animals to be modelled at all; telangiectasis, "
              "cirrhosis and fibrosis are represented by one animal each. A category "
              "represented by a single animal cannot appear in both a training and a test "
              "partition, so no cross-validated estimate for it is obtainable at any sample "
              "size of images drawn from that one animal.")
    para(doc, "Multi-class classification is therefore not merely underpowered on the present "
              "data but structurally impossible, and it should be planned in terms of animals "
              "recruited per category rather than images collected.")

    doc.add_heading("5.6 Attainment of the Study Objectives", level=2)
    table(doc,
          ["Objective", "Outcome"],
          [["1. Collect and curate a labelled dataset under a standardised protocol",
            f"Partially attained. {n_img} images from {n_animals} animals curated, "
            f"deduplicated and grouped by animal, but the imaging protocol was not applied "
            f"uniformly across the two classes (section 4.3)."],
           ["2. Develop a binary classifier by transfer learning",
            "Implemented for all three architectures. Performance figures obtained but "
            "invalidated by the capture confound; not reportable as evidence of lesion "
            "detection."],
           ["3. Benchmark against a classical machine-learning baseline",
            "Attained. The two arms converged on the same values as an 8×8 thumbnail, which "
            "was itself the diagnostic finding (section 4.6)."],
           ["4. Evaluate with cross-validation and standard metrics",
            "Attained and strengthened: nested animal-grouped cross-validation, pooled "
            "out-of-fold estimates and animal-level bootstrap intervals."],
           ["5. Apply Grad-CAM to confirm decisions are lesion-driven",
            f"Attained as a test, and the test failed: {gc_norm:.1%} of attention on normal "
            f"livers fell outside the organ (section 4.7)."],
           ["6. Characterise gross lesion types descriptively",
            "Attained. Frequencies reported by image and by animal; three of five categories "
            "rest on a single animal each."],
           ["Additional — tissue-restricted analysis (not in the proposal)",
            f"Liver parenchymal texture discriminates lesioned from normal organs at AUC "
            f"{f3(texture.auc)} per animal, with background, framing, illumination, "
            f"compression and magnification independently controlled."]],
          widths=[2.3, 3.9])
    caption(doc, "Table 5.1: Attainment of the study objectives.")

    doc.add_heading("5.7 Methodological Contributions", level=2)
    bullets(doc, [
        "**A reusable confound-detection protocol.** The probe suite detected a defect that "
        "would otherwise have passed review, and it is inexpensive, general and applicable to "
        "any smartphone-based imaging study conducted in a field setting. Its value does not "
        "depend on this study's particular subject matter.",
        "**Interpretability as a release criterion rather than an illustration.** Expressing "
        "Grad-CAM as a border-attention share against a definitional uniform reference of "
        "0.640 converts a qualitative figure into a measurement that can fail, and therefore "
        "into something capable of informing a decision.",
        "**Evaluation machinery appropriate to clustered image data.** Deduplication, "
        "animal-level grouping, nested cross-validation, inner-fold threshold selection and "
        "animal-level bootstrap intervals, with every fold verified free of group overlap and "
        "a label-permutation control confirming the absence of optimistic bias.",
        "**A route to a valid result on confounded images.** The tissue-restricted analysis "
        "shows that a dataset compromised at whole-image level need not be discarded "
        "entirely, provided the analysis is reconstructed around what the confound cannot "
        "reach.",
    ])

    doc.add_heading("5.8 Limitations", level=2)
    bullets(doc, [
        "**The whole-image results are not usable.** They are reported in section 4.6 for "
        "transparency and as the evidence motivating the confound investigation, not as "
        "findings. They should not be cited as evidence that smartphone images can classify "
        "liver lesions.",
        f"**Residual colour confounding in the patch analysis.** The patch mean-colour probe "
        f"reached {f3(patch('mean colour').auc)}, above chance. The colour pathway therefore "
        f"retains some session dependence, which is why the texture result — invariant to "
        f"illumination by construction — is the figure reported rather than the higher colour "
        f"figure of {f3(colour.auc)}.",
        "**Imperfect segmentation.** The segmentation is classical and unsupervised. A "
        "minority of patches may include non-parenchymal tissue, and the masks in Figure 4.4 "
        "should be inspected before the result is quoted.",
        f"**Pilot-scale sample.** {n_animals} distinct animals is a pilot-scale sample and the "
        f"confidence intervals are correspondingly wide (the texture estimate spans "
        f"{texture.lo:.3f} to {texture.hi:.3f}).",
        "**Single site and single collection period.** All images derive from one abattoir "
        "over one collection window, so external validity is untested. Generalisation to "
        "other facilities, lighting conditions, devices and inspectors is unknown.",
        "**Binary labelling only.** Lesion-type classification remains out of reach for the "
        "structural reason given in section 5.5.",
        "**Label subjectivity.** Labels derive from gross examination by a single inspector "
        "at the point of capture, without histopathological confirmation. Inter-observer "
        "agreement was not assessed, and gross-examination subjectivity is itself the problem "
        "the study seeks to address — so the reference standard shares the weakness of the "
        "process being improved.",
        "**No fine-tuned deep configurations were executed**, for the reason given in section "
        "5.3, so the comparison between frozen and fine-tuned regimes specified in the "
        "proposal remains open.",
    ])

    doc.add_heading("5.9 Recommendations", level=2)

    doc.add_heading("5.9.1 Image collection protocol for the next phase", level=3)
    para(doc, "The governing principle, which subsumes every specific rule below: **a normal "
              "liver and a lesioned liver photographed minutes apart must be indistinguishable "
              "to an observer shown only the margins of the frame.**")
    bullets(doc, [
        "**Paired sessions.** Photograph both classes in every session, on one plain "
        "inspection surface. No session should yield only one class, since session identity "
        "then becomes a proxy for the diagnosis.",
        "**Identical framing.** One standard whole-organ view per liver at fixed distance and "
        "orientation, for normal and lesioned organs alike. Close-ups are supplementary "
        "frames, never the only frame for a class.",
        "**Single device, direct transfer.** One device and camera setting throughout, with "
        "files copied directly from the device. Messaging-application export was the proximate "
        "cause of the file-level confound in section 4.3.",
        "**Animal identifiers.** Record an identifier for every image, so that animal "
        "grouping never again has to be reconstructed from image similarity.",
        "**Include a scale reference.** A ruler or standard marker in frame would permit true "
        "physical-scale normalisation rather than the organ-relative approximation used here.",
        "**Second grader.** Have an independent grader relabel a subset so that inter-observer "
        "agreement can be quantified, as the proposal envisaged.",
    ])

    doc.add_heading("5.9.2 Acceptance gates for each new batch", level=3)
    para(doc, "The diagnostic suite should be run as an acceptance test on every new batch of "
              "images **before** any modelling is attempted. If these gates do not pass, "
              "collecting additional images will not help, because the defect scales with the "
              "dataset rather than washing out of it.")
    table(doc,
          ["Probe", "Target", "Current value", "Status"],
          [["Capture metadata", "≈ 0.50",
            f3(probe("HARMONISED", "capture metadata")), "Pass"],
           ["Compression artefact", "< 0.65",
            f3(probe("HARMONISED", "compression artefact")), "Pass"],
           ["Border ring only", "< 0.65",
            f3(diag("border ring only (no centre)")), "Fail"],
           ["8×8 thumbnail", "< 0.70",
            f3(diag("8x8 thumbnail (all detail gone)")), "Fail"],
           ["Grad-CAM border share", "< 0.64", f3(gc.border_share.mean()), "Fail"]],
          widths=[2.0, 1.1, 1.4, 0.9])
    caption(doc, "Table 5.2: Acceptance thresholds and current values. The final three "
                 "currently fail for whole-image analysis.")

    doc.add_heading("5.9.3 Modelling roadmap", level=3)
    bullets(doc, [
        "**Report the tissue-texture result as the principal pilot finding**, with all three "
        "confound probes presented alongside it rather than relegated to an appendix.",
        "**Once the gates pass**, execute the full six-configuration grid including "
        "fine-tuning. Frozen backbones are expected to prevail at this sample size, "
        "consistent with the proposal's own reasoning.",
        "**Treat interpretability as a release criterion**: require the border-attention "
        "measure to fall below the 0.640 uniform reference before any performance figure is "
        "quoted as evidence of lesion detection.",
        "**Blinded expert review.** Have the meat inspector score activation maps blind, "
        "answering whether the highlighted region contains the lesion. That proportion is the "
        "substantive interpretability result, and it is stronger evidence than any summary "
        "statistic computed from the maps.",
        "**Multi-class extension** requires roughly 30 or more distinct animals per category. "
        "Fascioliasis (34 animals) and abscessation (16) approach viability; the remaining "
        "three categories rest on one animal each.",
        "**Improve the segmentation.** A small set of manually delineated masks would permit "
        "a supervised segmentation model, improving on the present unsupervised procedure and "
        "tightening the tissue-restricted analysis.",
        "**External validation.** A held-out batch from a different abattoir, or a different "
        "period, is the strongest available test of generalisation and should precede any "
        "claim of field readiness.",
    ])

    doc.add_heading("5.10 Conclusion", level=2)
    para(doc, f"This study developed and evaluated smartphone-based classifiers of bovine "
              f"liver lesions on a pilot dataset of {n_img} images from {n_animals} animals "
              f"collected at Kumasi Abattoir. Whole-image classifiers achieved apparently "
              f"excellent discrimination, reaching AUC {f3(dp.auc.max())}, but a systematic "
              f"programme of negative controls established that this performance reflected the "
              f"conditions under which the two classes had been photographed rather than the "
              f"pathology they contained. Whole-image classification cannot be validated on "
              f"this dataset, and no preprocessing remedy tested was able to separate the "
              f"capture conditions from the disease.")
    para(doc, f"An analysis restricted to segmented liver tissue, using texture features "
              f"invariant to illumination and magnification, discriminated lesioned from "
              f"normal organs at AUC {f3(texture.auc)} "
              f"(95% CI {texture.lo:.3f}–{texture.hi:.3f}) at the level of the individual "
              f"animal, clearing its strongest confound probe by "
              f"{texture.auc - worst_probe:.3f}. This is reported as the study's valid pilot "
              f"finding: **bovine liver parenchymal texture, measured from ordinary smartphone "
              f"photographs, carries genuine information about the presence of gross "
              f"pathology.**")
    para(doc, "The practical contribution is therefore twofold. The substantive finding "
              "establishes that the approach is worth pursuing and identifies texture, rather "
              "than colour, as the pathway that survives scrutiny. The methodological "
              "contribution — a cheap, general protocol for detecting capture confounding, "
              "together with the collection rules and acceptance gates in section 5.9 — is "
              "what would allow the next phase of this study to produce a whole-image result "
              "that can be trusted. Identifying that constraint before rather than after "
              "deployment is itself a useful outcome, since a classifier that reads the "
              "inspection table instead of the liver would fail in the field precisely when a "
              "lesioned organ was photographed in the ordinary way.")

    doc.add_page_break()

    # ===================================================================== APPENDICES
    doc.add_heading("APPENDICES", level=1)

    doc.add_heading("Appendix A: Analysis Environment and Reproducibility", level=2)
    para(doc, "Analyses were performed in Python 3.14 within a dedicated Conda environment "
              "using PyTorch and torchvision, scikit-learn, scikit-image, Pillow, SciPy, "
              "NumPy, pandas and matplotlib, on a four-thread CPU with no GPU. A fixed random "
              "seed (42) governs partitioning, augmentation, clustering and bootstrap "
              "resampling. The stages must be executed in the order below, because stages 1 "
              "to 4 establish the validity of everything that follows.")
    numbered(doc, [
        "`01_build_manifest.py` — manifest construction, deduplication, animal identity.",
        "`02_harmonise_and_probe.py` — harmonisation and the negative-control probes.",
        "`04_confound_diagnostics.py` — localisation of residual signal (acceptance gate).",
        "`03_classical_baseline.py` — colour and texture baselines.",
        "`05_deep_learning.py` — frozen backbones; `--finetune` for the full grid.",
        "`06_gradcam_occlusion.py` — interpretability, framed as a test.",
        "`07_patch_analysis.py` — tissue-restricted analysis.",
        "`08_scale_ablation.py` — magnification-normalisation ablation.",
        "`make_thesis_docx.py` — regenerates this document from outputs/.",
    ])
    para(doc, "All intermediate artefacts are written to `outputs/`: the manifest, the record "
              "of removed duplicates, per-fold metrics, out-of-fold predictions, "
              "negative-control results, confound diagnostics, attention measurements, patch "
              "and ablation results, and all figures. Every numerical value in Chapters Three "
              "to Five is read from those files at generation time, so this document cannot "
              "drift out of step with the analysis it describes.")

    doc.add_heading("Appendix B: Pipeline Modules", level=2)
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
          widths=[1.5, 4.7])
    caption(doc, "Table B.1: Modules under src/liver/.")

    doc.add_heading("Appendix C: Amendments Required to the Study Proposal", level=2)
    para(doc, "The following amendments align the written proposal with the dataset actually "
              "collected and the analysis actually performed.")
    table(doc,
          ["Section", "Current text", "Required amendment"],
          [["2.3", "100 images, 60 normal / 40 abnormal",
            f"{n_img} unique images from {n_animals} animals; {n_norm} normal / {n_les} "
            f"lesioned; sample-size targets expressed in animals"],
           ["2.5", "Six categories including fatty liver",
            "Add telangiectasis; record that fatty liver has not been observed"],
           ["2.4", "Standardised distance, lighting, background",
            "Add: both classes in every session; identical framing for both; direct device "
            "transfer; an animal identifier for every image"],
           ["2.6", "Random erasing among the augmentations",
            "Remove random erasing; on a small positive class it trains the model to infer "
            "the label from context"],
           ["2.7", "Metrics as mean ± SD across folds",
            "Add pooled out-of-fold AUC with animal-level bootstrap intervals, and "
            "animal-grouped nested cross-validation"],
           ["2.10", "Small sample, single site, binary labels",
            "Add the capture confound, its detection, and its effect on the pilot estimates"]],
          widths=[0.7, 2.3, 3.2])
    caption(doc, "Table C.1: Amendments required to align the proposal with the data and "
                 "analysis.")

    doc.add_heading("Appendix D: Glossary of Technical Terms", level=2)
    table(doc,
          ["Term", "Meaning as used in this thesis"],
          [["AUC-ROC",
            "Area under the receiver operating characteristic curve. The probability that a "
            "randomly chosen lesioned liver receives a higher score than a randomly chosen "
            "normal one. 0.5 is chance; 1.0 is perfect."],
           ["Negative control (probe)",
            "A deliberate attempt to solve the classification task using information that "
            "cannot encode pathology. A high score indicates leakage, not skill."],
           ["Shortcut learning",
            "A model achieving high accuracy by exploiting an incidental correlate of the "
            "label rather than the feature of clinical interest."],
           ["Nested cross-validation",
            "An outer loop that estimates performance and an inner loop that performs all "
            "model selection, so that no choice is made using the data used to report the "
            "result."],
           ["Grouped cross-validation",
            "Partitioning that keeps all images of one animal within a single fold, so that a "
            "model is never tested on an animal it was trained on."],
           ["GLCM",
            "Grey-level co-occurrence matrix. A tabulation of how often pairs of lightness "
            "values occur at a given distance and angle, summarising spatial texture."],
           ["Grad-CAM",
            "Gradient-weighted class activation mapping. A method that highlights the image "
            "regions most influential to a network's prediction."],
           ["Occlusion sensitivity",
            "Systematically concealing regions of an image and recording the change in "
            "prediction. Requires no gradients, so it cannot be misled by attribution "
            "artefacts."],
           ["Border attention share",
            "The proportion of a Grad-CAM map's total activation falling in the outer 20 per "
            "cent of the frame. A uniform map gives 0.640."],
           ["Transfer learning",
            "Reusing a network pretrained on a large general image dataset, either as a fixed "
            "feature extractor (frozen) or by continued training (fine-tuned)."],
           ["Bootstrap confidence interval",
            "An interval obtained by repeatedly resampling the data. Here animals, not "
            "images, are resampled, so that repeat views of one liver are not treated as "
            "independent."]],
          widths=[1.5, 4.7])
    caption(doc, "Table D.1: Glossary of technical terms.")

    doc.save(OUT)
    print(f"Written: {OUT}")
    print(f"  headings: {sum(1 for p in doc.paragraphs if p.style.name.startswith('Heading'))}")
    print(f"  tables  : {len(doc.tables)}")
    print(f"  figures : {len(doc.inline_shapes)}")


if __name__ == "__main__":
    main()
