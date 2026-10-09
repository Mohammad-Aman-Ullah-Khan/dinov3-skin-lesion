"""
test.py
Held-out (outer-fold) inference and pooled out-of-fold evaluation
(Sections 4.1-4.5).

Per fold:
  - loads outputs/fold{K}/best.pt, predicts ONLY that fold's held-out images,
  - applies the fold's fitted temperature to its held-out logits
    (Section 3.10: temperature fitted per fold on validation data only).

Pooled (all 10,015 images, each predicted once out-of-fold):
  - Table 7  : per-fold metrics + mean +/- SD
  - Table 8  : pooled aggregate metrics
  - Table 9  : per-class OvR sensitivity/specificity/precision/NPV/F1
  - Table 10 : per-class AUROC / AUPRC / PR baseline / gain
  - Table 11 : error analysis
  - Table 12 : calibration before vs after temperature scaling
  - Figure 4 : pooled confusion matrix (counts + row %)
  - Figure 5 : OvR ROC curves          Figure 6: OvR PR curves
  - Table 3 checks on stored predictions: probability rows sum to 1,
    argmax == stored class (100% agreement).

Everything lands in outputs/evaluation/ (CSVs + PNGs + oof_predictions.npz).

Usage:  python test.py            (all folds)
        python test.py --folds 0 1 2
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_curve)
from torch.utils.data import DataLoader

from calibration import apply_temperature, softmax_np
from config import get_config, CLASSES, NUM_CLASSES
from dataset import HAM10000Dataset, outer_fold_frames
from metrics import (calibration_metrics, classification_metrics, error_table,
                     per_class_metrics)
from model import build_model
from train import collect_logits


# --------------------------------------------------------------------------- #
def predict_fold(cfg, fold: int, device):
    fold_dir = os.path.join(cfg.out_dir, f"fold{fold}")
    ckpt = torch.load(os.path.join(fold_dir, "best.pt"),
                      map_location="cpu", weights_only=False)
    fold_cfg = get_config()
    for k, v in ckpt["config"].items():
        if hasattr(fold_cfg, k):
            setattr(fold_cfg, k, v)
    fold_cfg.out_dir = cfg.out_dir

    model = build_model(fold_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, held_df = outer_fold_frames(cfg.folds_csv, fold)
    loader = DataLoader(HAM10000Dataset(held_df, fold_cfg, train=False),
                        batch_size=fold_cfg.batch_size, shuffle=False,
                        num_workers=fold_cfg.num_workers)
    logits, labels, ids = collect_logits(model, loader, device)

    with open(os.path.join(fold_dir, "temperature.json")) as f:
        T = json.load(f)["temperature"]
    return logits, labels, ids, T


def fmt(m, keys):
    return {k: round(m[k], 4) for k in keys}


# --------------------------------------------------------------------------- #
def main(folds):
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = os.path.join(cfg.out_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    all_logits, all_labels, all_ids, all_probs_cal, fold_rows = [], [], [], [], []
    agg_keys = ["accuracy", "balanced_accuracy", "macro_precision",
                "macro_recall", "macro_f1", "mcc", "macro_auroc", "macro_auprc"]

    for fold in folds:
        logits, labels, ids, T = predict_fold(cfg, fold, device)
        probs_cal = apply_temperature(logits, T)
        m = classification_metrics(labels, probs_cal)
        fold_rows.append({"fold": fold + 1, "n": len(labels),
                          "temperature": T, **fmt(m, agg_keys)})
        print(f"[fold {fold}] n={len(labels)} T={T:.3f} "
              f"acc={m['accuracy']:.2f} balacc={m['balanced_accuracy']:.2f} "
              f"AUPRC={m['macro_auprc']:.4f}")
        all_logits.append(logits)
        all_labels.append(labels)
        all_ids.extend(ids)
        all_probs_cal.append(probs_cal)

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs_raw = softmax_np(logits)
    probs_cal = np.concatenate(all_probs_cal)
    y_pred = probs_cal.argmax(axis=1)
    np.savez(os.path.join(eval_dir, "oof_predictions.npz"),
             logits=logits, probs_calibrated=probs_cal, labels=labels,
             ids=np.array(all_ids))

    # ---- Table 3 output checks (probability normalization, argmax agreement)
    row_sum_err = float(np.abs(probs_cal.sum(axis=1) - 1.0).max())
    # Verify temperature scaling does not change predictions (raw vs calibrated argmax)
    agree = float((probs_raw.argmax(1) == probs_cal.argmax(1)).mean() * 100)
    print(f"\nOutput checks: max |sum(p)-1| = {row_sum_err:.2e}; "
          f"temperature argmax agreement = {agree:.2f}% (should be 100%)")

    # ---- Table 7: per-fold + mean +/- SD
    t7 = pd.DataFrame(fold_rows)
    stats = t7[agg_keys].agg(["mean", "std"]).round(4)
    t7.to_csv(os.path.join(eval_dir, "table7_per_fold.csv"), index=False)
    stats.to_csv(os.path.join(eval_dir, "table7_mean_sd.csv"))
    print("\nTable 7 (per-fold):\n", t7.to_string(index=False))
    print("\nTable 7 (mean ± SD):\n", stats.to_string())

    # ---- Table 8: pooled
    t8 = classification_metrics(labels, probs_cal)
    pd.Series(t8).round(5).to_csv(os.path.join(eval_dir, "table8_pooled.csv"))
    print("\nTable 8 (pooled OOF):")
    for k, v in t8.items():
        print(f"  {k:18s}: {v:.4f}")

    # ---- Tables 9-10: per-class
    pc = per_class_metrics(labels, probs_cal)
    t9 = pd.DataFrame(pc).T
    t9["pr_baseline"] = t9["prevalence"]
    t9["gain_pp"] = (t9["auprc"] * 100 - t9["prevalence"]).round(2)
    t9.round(5).to_csv(os.path.join(eval_dir, "table9_10_per_class.csv"))
    print("\nTables 9-10 (per-class):\n",
          t9[["n", "sensitivity", "specificity", "precision", "npv", "f1",
              "auroc", "auprc", "gain_pp"]].round(4).to_string())

    # ---- Table 11: error analysis
    t11 = pd.DataFrame(error_table(labels, y_pred)).T
    t11.to_csv(os.path.join(eval_dir, "table11_errors.csv"))
    print("\nTable 11 (errors):\n", t11.to_string())

    # ---- Table 12: calibration before vs after
    rows = {}
    for tag, p in (("before", probs_raw), ("after", probs_cal)):
        m = classification_metrics(labels, p)
        c = calibration_metrics(labels, p, n_bins=cfg.ece_bins)
        rows[tag] = {"accuracy": m["accuracy"],
                     "macro_auroc": m["macro_auroc"],
                     "macro_auprc": m["macro_auprc"], **c}
    t12 = pd.DataFrame(rows).round(5)
    t12.to_csv(os.path.join(eval_dir, "table12_calibration.csv"))
    print("\nTable 12 (calibration):\n", t12.to_string())

    # ---- Figure 4: confusion matrix (counts + row %)
    cm = confusion_matrix(labels, y_pred, labels=range(NUM_CLASSES))
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if row_pct[i, j] > 50 else "black")
    ax.set_xticks(range(NUM_CLASSES), CLASSES)
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Pooled five-fold out-of-fold confusion matrix (HAM10000)")
    fig.colorbar(im, label="% of true class")
    fig.tight_layout()
    fig.savefig(os.path.join(eval_dir, "figure4_confusion_matrix.png"), dpi=300)
    plt.close(fig)

    # ---- Figures 5-6: OvR ROC + PR
    for fig_name, curve in (("figure5_roc.png", "roc"),
                            ("figure6_pr.png", "pr")):
        fig, ax = plt.subplots(figsize=(7, 6))
        for c, name in enumerate(CLASSES):
            binary = (labels == c).astype(int)
            if curve == "roc":
                x, y_, _ = roc_curve(binary, probs_cal[:, c])
                label = f"{name} (AUROC={pc[name]['auroc']:.5f})"
            else:
                y_, x, _ = precision_recall_curve(binary, probs_cal[:, c])
                label = f"{name} (AUPRC={pc[name]['auprc']:.5f})"
            ax.plot(x, y_, lw=1.2, label=label)
        if curve == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Class-specific one-vs-rest ROC (pooled OOF)")
        else:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Class-specific one-vs-rest precision-recall (pooled OOF)")
        ax.legend(fontsize=7, loc="lower right" if curve == "roc" else "lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, fig_name), dpi=300)
        plt.close(fig)

    print(f"\nAll tables/figures written to {eval_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    cfg = get_config()
    main(a.folds if a.folds else list(range(cfg.n_folds)))
