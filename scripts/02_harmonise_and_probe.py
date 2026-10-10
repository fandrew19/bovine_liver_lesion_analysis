"""Phase 3 -- harmonise every image, then measure what leakage remains.

Runs the negative controls on the raw images and again on the harmonised ones.
The raw numbers evidence finding F-01; the harmonised numbers are the gate that
must pass before any model is trained.
"""
import _bootstrap  # noqa: F401

import warnings

import pandas as pd

from liver import harmonise, probes
from liver.config import (
    DATA_ROOT,
    HARMONISED_ROOT,
    MANIFEST_CSV,
    OUTPUTS,
    PROJECT_ROOT,
    ensure_dirs,
)

PROBE_FAMILIES = [
    ("capture metadata", probes.capture_metadata_features, True),
    ("compression artefact", probes.compression_artefact_features, True),
    ("image content stats", probes.content_statistics_features, False),
]


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    manifest = pd.read_csv(MANIFEST_CSV)
    y = manifest["label"].to_numpy()
    groups = manifest["liver_id"].to_numpy()

    print(f"Harmonising {len(manifest)} images -> {HARMONISED_ROOT.name}/ (PNG) ...")
    manifest = harmonise.build_harmonised_set(manifest, DATA_ROOT, force=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print("  done; manifest updated with harmonised_path\n")

    stages = {
        "RAW": [DATA_ROOT / p for p in manifest["path"]],
        "HARMONISED": [PROJECT_ROOT / p for p in manifest["harmonised_path"]],
    }

    rows = []
    for stage, paths in stages.items():
        for name, extract, gated in PROBE_FAMILIES:
            row = probes.run_probe(
                f"[{stage}] {name}", extract(paths), y, groups, gated=gated
            )
            rows.append(row)
            print(f"  {row['probe']:38s} AUC={row['worst_auc']:.3f}   {row['verdict']}")

    rows.append(
        probes.shuffled_label_probe(
            probes.compression_artefact_features(stages["HARMONISED"]), y, groups
        )
    )
    print(f"  {rows[-1]['probe']:38s} AUC={rows[-1]['worst_auc']:.3f}   {rows[-1]['verdict']}")

    df = probes.report(rows)
    out = OUTPUTS / "negative_controls.csv"
    df.to_csv(out, index=False)
    print(f"\nWritten to {out.relative_to(PROJECT_ROOT)}\n")
    print(df.to_string(index=False))

    failed = df[
        df["gated"] & df["probe"].str.startswith("[HARMONISED]")
        & (df["verdict"] == "LEAKS")
    ]["probe"].tolist()
    print()
    if failed:
        print("GATE FAILED -- still recoverable after harmonisation:")
        for p in failed:
            print(f"    {p}")
    else:
        print(f"GATE PASSED -- no gated probe reaches AUC {probes.GATE_AUC} "
              "after harmonisation.")
        print("Residual content-statistic skill is expected and is not leakage;")
        print("Grad-CAM and the occlusion test (script 05) check it is lesion-driven.")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
