"""
external_validation.py
External evaluation (Section 4.8 / Table 15) on ISIC 2017, PAD-UFES-20 and
BCN20000. External data are NEVER used for training, HSO, checkpoint
selection, or calibration (Table 3 "External data isolation").

Label harmonization is the scientifically sensitive step and must be stated
explicitly in the manuscript (the Abstract already concedes dependence on
"label harmonization"). This script expects a prepared CSV per dataset with
columns:
    image_path,class          # class in {MEL,NV,BCC,BKL,AKIEC,DF,VASC}
Build those CSVs with the helper functions below, and REPORT in the paper:
  - which source labels were mapped to which of the 7 classes,
  - which images/classes were EXCLUDED because no honest mapping exists
    (e.g., ISIC 2017 has only 3 categories; PAD-UFES-20 has SCC and no DF/VASC
    equivalents in some releases; BCN20000 contains SCC).
  - whether metrics are macro over 7 classes or over the classes present.
    (metrics.py computes macro over classes PRESENT in y_true; a 7-class macro
    on a 3-class dataset is undefined — a reviewer will check this.)

Prediction ensemble: mean of the 5 fold models' calibrated probabilities
(each fold's own temperature). If the manuscript used a single model instead,
run with --fold K and state which fold.

Usage:
    python external_validation.py --csv data/external/isic2017.csv --name ISIC2017
    python external_validation.py --csv data/external/padufes20.csv --name PAD-UFES-20 --fold 0
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from calibration import apply_temperature
from config import get_config, CLASSES
from dataset import ExternalDataset
from metrics import classification_metrics, per_class_metrics
from model import build_model
from train import collect_logits


# ---------------------------------------------------------------------------
# Harmonization helpers — EDIT to match the exact release you downloaded, and
# document every choice in the manuscript.
# ---------------------------------------------------------------------------
def build_isic2017_csv(gt_csv: str, image_dir: str, out_csv: str):
    """ISIC 2017 ground truth: melanoma / seborrheic_keratosis / nevus.
    Mapping: melanoma->MEL, seborrheic_keratosis->BKL, remainder->NV.
    Only 3 of 7 classes exist — macro metrics are over these 3."""
    df = pd.read_csv(gt_csv)
    rows = []
    for _, r in df.iterrows():
        if r.get("melanoma", 0) == 1:
            cls = "MEL"
        elif r.get("seborrheic_keratosis", 0) == 1:
            cls = "BKL"
        else:
            cls = "NV"
        rows.append({"image_path": os.path.join(image_dir,
                                                r["image_id"] + ".jpg"),
                     "class": cls})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} rows; classes: MEL/BKL/NV only)")


def build_padufes20_csv(metadata_csv: str, image_dir: str, out_csv: str):
    """PAD-UFES-20 diagnostics: BCC, SCC, ACK, SEK, MEL, NEV.
    Mapping: BCC->BCC, ACK->AKIEC, SEK->BKL, MEL->MEL, NEV->NV.
    SCC is EXCLUDED (no honest 7-class target) — report the exclusion count."""
    mapping = {"BCC": "BCC", "ACK": "AKIEC", "SEK": "BKL",
               "MEL": "MEL", "NEV": "NV"}
    df = pd.read_csv(metadata_csv)
    kept, dropped = [], 0
    for _, r in df.iterrows():
        dx = str(r["diagnostic"]).upper()
        if dx not in mapping:
            dropped += 1
            continue
        kept.append({"image_path": os.path.join(image_dir, r["img_id"]),
                     "class": mapping[dx]})
    pd.DataFrame(kept).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(kept)} rows; EXCLUDED {dropped} images "
          f"(e.g., SCC) — report this in the manuscript)")


def build_bcn20000_csv(metadata_csv: str, image_dir: str, out_csv: str):
    """BCN20000 (ISIC 2019 style dx). SCC is EXCLUDED for the same reason."""
    mapping = {"mel": "MEL", "nv": "NV", "bcc": "BCC", "bkl": "BKL",
               "ak": "AKIEC", "akiec": "AKIEC", "df": "DF", "vasc": "VASC"}
    df = pd.read_csv(metadata_csv)
    kept, dropped = [], 0
    for _, r in df.iterrows():
        dx = str(r["diagnosis"] if "diagnosis" in df.columns else r["dx"]).lower()
        if dx not in mapping:
            dropped += 1
            continue
        img = r["isic_id"] if "isic_id" in df.columns else r["image"]
        kept.append({"image_path": os.path.join(image_dir, f"{img}.jpg"),
                     "class": mapping[dx]})
    pd.DataFrame(kept).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(kept)} rows; EXCLUDED {dropped} images — "
          f"report this in the manuscript)")


# ---------------------------------------------------------------------------
def evaluate_external(csv_path: str, name: str, single_fold: int = None):
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ExternalDataset(csv_path, cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    folds = [single_fold] if single_fold is not None else list(range(cfg.n_folds))
    probs_sum, labels = None, None
    for fold in folds:
        fold_dir = os.path.join(cfg.out_dir, f"fold{fold}")
        ckpt = torch.load(os.path.join(fold_dir, "best.pt"),
                          map_location="cpu", weights_only=False)
        fold_cfg = get_config()
        for k, v in ckpt["config"].items():
            if hasattr(fold_cfg, k):
                setattr(fold_cfg, k, v)
        model = build_model(fold_cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        logits, labels, _ = collect_logits(model, loader, device)
        with open(os.path.join(fold_dir, "temperature.json")) as f:
            T = json.load(f)["temperature"]
        p = apply_temperature(logits, T)
        probs_sum = p if probs_sum is None else probs_sum + p
        print(f"  fold {fold} done (T={T:.3f})")
    probs = probs_sum / len(folds)

    present = sorted(set(labels.tolist()))
    print(f"\n[{name}] n={len(labels)}; classes present: "
          f"{[CLASSES[c] for c in present]} "
          f"(macro metrics computed over PRESENT classes)")
    m = classification_metrics(labels, probs)
    for k, v in m.items():
        print(f"  {k:18s}: {v:.4f}")

    out = os.path.join(cfg.out_dir, "evaluation",
                       f"external_{name.replace(' ', '_')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.Series(m).round(5).to_csv(out)
    pc = pd.DataFrame(per_class_metrics(labels, probs)).T
    pc.round(5).to_csv(out.replace(".csv", "_per_class.csv"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="prepared external CSV (image_path,class)")
    ap.add_argument("--name", default="external")
    ap.add_argument("--fold", type=int, default=None,
                    help="use one fold model instead of the 5-fold mean")
    ap.add_argument("--build", choices=["isic2017", "padufes20", "bcn20000"])
    ap.add_argument("--src-csv", help="source metadata/ground-truth CSV")
    ap.add_argument("--image-dir", help="source image directory")
    ap.add_argument("--out-csv", help="harmonized CSV to write")
    a = ap.parse_args()

    if a.build:
        builder = {"isic2017": build_isic2017_csv,
                   "padufes20": build_padufes20_csv,
                   "bcn20000": build_bcn20000_csv}[a.build]
        builder(a.src_csv, a.image_dir, a.out_csv)
    else:
        evaluate_external(a.csv, a.name, a.fold)
