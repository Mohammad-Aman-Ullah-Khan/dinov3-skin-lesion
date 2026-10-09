"""
validate.py
Evaluate a trained fold checkpoint on its INNER VALIDATION split (the 10%
lesion-grouped subset used for model selection, HSO and temperature fitting —
Section 3.10). This is a development-side check only; held-out performance
comes from test.py.

Usage:
    python validate.py --fold 0
    python validate.py --fold 0 --calibrated     # apply the fitted temperature
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration import apply_temperature, softmax_np
from config import get_config, CLASSES
from dataset import HAM10000Dataset, inner_split, outer_fold_frames
from metrics import calibration_metrics, classification_metrics, per_class_metrics
from model import build_model
from train import collect_logits


def main(fold: int, calibrated: bool, out_dir: str = None):
    cfg = get_config(**({"out_dir": out_dir} if out_dir else {}))
    fold_dir = os.path.join(cfg.out_dir, f"fold{fold}")
    ckpt = torch.load(os.path.join(fold_dir, "best.pt"),
                      map_location="cpu", weights_only=False)
    for k, v in ckpt["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    dev_df, _ = outer_fold_frames(cfg.folds_csv, fold)
    _, va_df = inner_split(dev_df, cfg.inner_val_fraction, cfg.split_seed)
    loader = DataLoader(HAM10000Dataset(va_df, cfg, train=False),
                        batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    logits, labels, _ = collect_logits(model, loader, device)
    if calibrated:
        with open(os.path.join(fold_dir, "temperature.json")) as f:
            T = json.load(f)["temperature"]
        probs = apply_temperature(logits, T)
        print(f"Applying fitted temperature T={T:.3f}")
    else:
        probs = softmax_np(logits)

    m = classification_metrics(labels, probs)
    c = calibration_metrics(labels, probs, n_bins=cfg.ece_bins)
    print(f"\n[fold {fold}] inner-validation "
          f"({'calibrated' if calibrated else 'uncalibrated'})")
    for k, v in {**m, **c}.items():
        print(f"  {k:18s}: {v:.4f}")

    print("\nPer-class (one-vs-rest):")
    for name, row in per_class_metrics(labels, probs).items():
        print(f"  {name:6s} n={row['n']:5d} sens={row['sensitivity']:6.2f} "
              f"spec={row['specificity']:6.2f} prec={row['precision']:6.2f} "
              f"f1={row['f1']:6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--calibrated", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.fold, a.calibrated, a.out_dir)
