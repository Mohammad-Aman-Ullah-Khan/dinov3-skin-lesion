"""
ablation.py
Incremental component contribution analysis (Section 4.7 / Table 14).

Rows, trained under the identical protocol (same folds, same inner splits,
same schedule):

  1. backbone            DINOv3 ConvNeXt-B, deepest feature, linear head, CE
  2. +LADA               LADA on all levels, deepest feature, linear head, CE
  3. +ADSF               LADA + ADSF, deepest feature, linear head, CE
  4. +CSGA               LADA + ADSF + CSGA, linear head, CE
  5. +adaptive margin    full architecture, cosine head, margin only (no ω_c)
  6. full model          full architecture, margin + effective-number weights

Each row runs train.py's machinery for every fold into its own output
directory; evaluate any row with:  python test.py  after pointing
Config.out_dir at that row's directory (or pass --out-dir here and to test.py).

Usage:
    python ablation.py --row backbone --all
    python ablation.py --row full --fold 0
    python ablation.py --list
"""

import argparse

from config import get_config
from train import run_fold

ROWS = {
    "backbone": dict(use_lada=False, use_adsf=False, use_csga=False,
                     head="linear_ce"),
    "lada": dict(use_lada=True, use_adsf=False, use_csga=False,
                 head="linear_ce"),
    "adsf": dict(use_lada=True, use_adsf=True, use_csga=False,
                 head="linear_ce"),
    "csga": dict(use_lada=True, use_adsf=True, use_csga=True,
                 head="linear_ce"),
    "margin": dict(use_lada=True, use_adsf=True, use_csga=True,
                   head="cb_adaptive_margin", use_margin=True,
                   use_class_weights=False),
    "full": dict(use_lada=True, use_adsf=True, use_csga=True,
                 head="cb_adaptive_margin", use_margin=True,
                 use_class_weights=True),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", choices=list(ROWS), default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hso", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or a.row is None:
        for name, kw in ROWS.items():
            print(f"{name:10s} -> {kw}")
        raise SystemExit(0)

    overrides = dict(ROWS[a.row])
    overrides["out_dir"] = f"outputs/ablation_{a.row}"
    folds = range(get_config().n_folds) if a.all else [a.fold or 0]
    for f in folds:
        run_fold(get_config(**overrides), f, use_hso=a.hso)

# ---------------------------------------------------------------------------
# Table 13 (reproduced baselines) note:
# ResNet50 / ConvNeXt V2-B / Swin V2-B / DINOv2 ViT-B/14 baselines are trained
# with the same folds, preprocessing, schedule and CE loss by swapping the
# backbone. The simplest faithful route: run ablation row "backbone" but with
#   Config.backbone_source = "timm" and Config.timm_backbone in
#   {"resnet50", "convnextv2_base", "swinv2_base_window8_256",
#    "vit_base_patch14_dinov2"}
# adjusting img_size where the architecture requires it, and documenting any
# such deviation in the manuscript. The PanDerm comparator requires the
# official PanDerm weights and license — it cannot be reproduced from this
# repository alone and MUST cite the exact checkpoint used.
# ---------------------------------------------------------------------------
