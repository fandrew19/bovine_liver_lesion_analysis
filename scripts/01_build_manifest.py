"""Phase 2 -- build the manifest, drop duplicates, assign liver identities.

Writes outputs/manifest.csv (the single source of truth for every later stage) and
outputs/dropped_duplicates.csv (an audit trail of what was removed and why).
"""
import _bootstrap  # noqa: F401

import pandas as pd

from liver import manifest
from liver.config import DATA_ROOT, DROPPED_CSV, MANIFEST_CSV, ensure_dirs


def main() -> None:
    ensure_dirs()
    print(f"Scanning {DATA_ROOT} ...")
    df, dropped = manifest.build(DATA_ROOT)

    df.to_csv(MANIFEST_CSV, index=False)
    dropped.to_csv(DROPPED_CSV, index=False)

    print(f"\nManifest written to {MANIFEST_CSV.relative_to(DATA_ROOT.parent)}")
    print(f"  images kept      : {len(df)}")
    print(f"  duplicates dropped: {len(dropped)}")
    print(f"  distinct livers  : {df['liver_id'].nunique()}")

    print("\n--- class balance ---")
    print(df["label"].map({0: "normal", 1: "abnormal"}).value_counts().to_string())

    print("\n--- lesion types (images / distinct livers) ---")
    by_type = df.groupby("lesion_type").agg(
        images=("path", "size"), livers=("liver_id", "nunique")
    ).sort_values("images", ascending=False)
    print(by_type.to_string())

    print("\n--- livers contributing more than one image ---")
    sizes = df.groupby("liver_id").size()
    multi = sizes[sizes > 1].sort_values(ascending=False)
    print(f"  {len(multi)} livers cover {int(multi.sum())} images")
    for liver_id, n in multi.head(12).items():
        names = ", ".join(df.loc[df.liver_id == liver_id, "path"].map(
            lambda p: p.rsplit("/", 1)[-1]))
        print(f"    {liver_id} ({n}): {names}")

    if len(dropped):
        print("\n--- dropped byte-identical duplicates ---")
        for _, row in dropped.iterrows():
            print(f"    {row['path']}  (md5 {row['md5'][:8]})")

    unnumbered = df[df["case_id"].isna() & (df["label"] == 1)]
    if len(unnumbered):
        print("\n--- lesion images with no case number in the filename ---")
        print("    (grouped by appearance only; confirm these are separate animals)")
        for _, row in unnumbered.iterrows():
            print(f"    {row['path']}  -> {row['liver_id']}")


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
