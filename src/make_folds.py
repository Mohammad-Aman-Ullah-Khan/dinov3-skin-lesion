"""
make_folds.py
Five-fold LESION-GROUPED cross-validation split for HAM10000 (Sections 3.1-3.2).

- Fold assignment at the lesion level (lesion_id), never image level.
- Stratified by diagnosis so per-fold class composition matches Table 2.
- Seed = 42 (Table 5).
- Emits outputs/folds.csv  (image_id, lesion_id, dx, class_idx, fold)
- Runs and prints the Table 3 leakage-control verification:
    * 0 overlapping lesion IDs between folds
    * fold accounting sums to the dataset total
    * per-class fold totals reproduce the dataset totals exactly
    * exact-duplicate control across folds (MD5 over image bytes)

Usage:
    python make_folds.py --metadata data/HAM10000/HAM10000_metadata.csv \
                         --image-dir data/HAM10000/images --out outputs/folds.csv
"""

import argparse
import hashlib
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import CLASSES, CLASS_TO_IDX, DX_TO_CLASS, get_config


def find_image(image_dir: str, image_id: str) -> str:
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(image_dir, image_id + ext)
        if os.path.exists(p):
            return p
    # HAM10000 is often shipped as two part-folders
    for sub in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        p = os.path.join(image_dir, sub, image_id + ".jpg")
        if os.path.exists(p):
            return p
    return ""


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_folds(metadata_csv: str, image_dir: str, out_csv: str,
               n_folds: int = 5, seed: int = 42, check_duplicates: bool = True):
    df = pd.read_csv(metadata_csv)
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")

    df = df.copy()
    df["class"] = df["dx"].str.lower().map(DX_TO_CLASS)
    if df["class"].isna().any():
        bad = df.loc[df["class"].isna(), "dx"].unique()
        raise ValueError(f"Unmapped dx labels: {bad}")
    df["class_idx"] = df["class"].map(CLASS_TO_IDX)

    # ---------------- lesion-grouped stratified assignment -------------------
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, held_idx) in enumerate(
        sgkf.split(df, y=df["class_idx"], groups=df["lesion_id"])
    ):
        df.loc[df.index[held_idx], "fold"] = fold
    assert (df["fold"] >= 0).all()

    # ---------------- Table 3 verification -----------------------------------
    print("=" * 72)
    print("Table 3 — Data partitioning and leakage control verification")
    print("=" * 72)

    # 1. Lesion group separation
    lesion_folds = df.groupby("lesion_id")["fold"].nunique()
    n_overlap = int((lesion_folds > 1).sum())
    print(f"Lesion group separation : {n_overlap} overlapping lesion IDs "
          f"({'PASS' if n_overlap == 0 else 'FAIL'})")

    # 2. Fold accounting
    total = len(df)
    fold_sizes = df["fold"].value_counts().sort_index()
    print(f"Fold accounting         : {fold_sizes.sum()} / {total} "
          f"({'PASS' if fold_sizes.sum() == total else 'FAIL'})")
    for f, n in fold_sizes.items():
        print(f"    Fold {f + 1}: {n} images")

    # 3. Class accounting (per-fold composition — compare with Table 2)
    table2 = df.pivot_table(index="fold", columns="class", values="image_id",
                            aggfunc="count", fill_value=0)[CLASSES]
    print("\nPer-fold class composition (compare with manuscript Table 2):")
    print(table2.to_string())
    col_ok = (table2.sum(axis=0).values ==
              df["class"].value_counts().reindex(CLASSES).values).all()
    print(f"Class accounting        : {'Exact match (PASS)' if col_ok else 'FAIL'}")
    # Warn if any fold is missing a class entirely — build_loss will assert on this
    zero_cells = [(f"fold {r+1}", c) for r in range(len(table2))
                  for c in CLASSES if table2.loc[r, c] == 0]
    if zero_cells:
        print(f"\n  WARNING: {len(zero_cells)} fold/class cell(s) have 0 images: "
              f"{zero_cells}")
        print("  This will cause build_loss() to fail. Consider --n-folds 3 "
              "or merging rare lesion groups.")
    else:
        print("Zero-class check        : No empty fold/class cells (PASS)")

    # 4. Exact-duplicate control across folds
    if check_duplicates:
        print("\nExact duplicate control (MD5 over image bytes) ...")
        hash_to_folds = defaultdict(set)
        n_missing = 0
        for _, row in df.iterrows():
            p = find_image(image_dir, row["image_id"])
            if not p:
                n_missing += 1
                continue
            hash_to_folds[md5_of(p)].add(int(row["fold"]))
        cross = sum(1 for folds in hash_to_folds.values() if len(folds) > 1)
        print(f"Exact duplicate control : {cross} cross-fold duplicates "
              f"({'PASS' if cross == 0 else 'FAIL — merge duplicate lesions and re-split'})")
        if n_missing:
            print(f"  WARNING: {n_missing} image files not found under {image_dir}; "
                  f"duplicate check incomplete.")
    else:
        print("Exact duplicate control : SKIPPED (--no-dup-check)")

    print(f"Random seeds            : split = {seed}")
    print("=" * 72)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df[["image_id", "lesion_id", "dx", "class", "class_idx", "fold"]].to_csv(
        out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=cfg.metadata_csv)
    ap.add_argument("--image-dir", default=cfg.image_dir)
    ap.add_argument("--out", default=cfg.folds_csv)
    ap.add_argument("--n-folds", type=int, default=cfg.n_folds)
    ap.add_argument("--seed", type=int, default=cfg.split_seed)
    ap.add_argument("--no-dup-check", action="store_true")
    a = ap.parse_args()
    make_folds(a.metadata, a.image_dir, a.out, a.n_folds, a.seed,
               check_duplicates=not a.no_dup_check)
