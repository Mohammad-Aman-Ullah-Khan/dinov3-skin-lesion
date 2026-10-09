"""
train.py
Outer-fold training exactly as Section 3.10 / Table 5:

  - 5 outer lesion-grouped folds; per fold: 10% lesion-grouped inner validation.
  - Full end-to-end fine-tuning; AdamW (0.90, 0.999, eps 1e-8); parameter-group
    LRs (backbone 2e-5 ref, new modules 2e-4 ref); weight decay 0.05 with
    bias/norm excluded; 5-epoch linear warmup -> cosine decay to 1e-6;
    <=60 epochs; batch 32; grad-norm clip 1.0; FP16 AMP; embedding dropout 0.2.
  - Checkpoint selection: inner-validation macro AUPRC (macro F1 tiebreak).
  - Early stopping: patience 10, min improvement 1e-4.
  - Optional nested HSO (hso.py) over {lr_backbone, lr_modules, weight_decay,
    dropout}; the outer held-out fold is never touched by HSO or selection.
  - Temperature scaling fitted on inner-validation logits (calibration.py).

Per fold this writes to  outputs/fold{K}/ :
    best.pt, config.json, hso_history.json (if HSO ran), temperature.json,
    val_logits.npz, train_log.csv

Usage:
    python train.py --fold 0                       # one fold, reference config
    python train.py --all                          # folds 0..4
    python train.py --all --hso --hso-epochs 12    # nested HSO, 12-epoch proxies
    python train.py --all --skip-hso               # explicit reference config
Ablation rows (Table 14) are launched from ablation.py.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from calibration import fit_temperature
from config import get_config, Config
from dataset import HAM10000Dataset, inner_split, outer_fold_frames
from losses import build_loss
from metrics import classification_metrics
from model import build_model


# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lambda(epoch: int, warmup: int, total: int,
                         base_lr: float, min_lr: float) -> float:
    """Multiplicative factor: linear warmup then cosine decay to min_lr."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    t = (epoch - warmup) / max(total - warmup, 1)
    floor = min_lr / base_lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels, ids = [], [], []
    for x, y, iid in loader:
        with torch.autocast(device_type=device.type, enabled=False):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
        ids.extend(iid)
    return np.concatenate(logits), np.concatenate(labels), ids


def evaluate_probs(logits: np.ndarray, labels: np.ndarray):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return classification_metrics(labels, probs)


# --------------------------------------------------------------------------- #
def train_one_config(cfg: Config, tr_df: pd.DataFrame, va_df: pd.DataFrame,
                     device, max_epochs: int, log_path: str = None,
                     save_best_to: str = None):
    """Train once, select best checkpoint by val macro AUPRC (F1 tiebreak).
    Returns (best_auprc, best_f1, best_state or None)."""
    set_seed(cfg.train_seed)
    train_ds = HAM10000Dataset(tr_df, cfg, train=True)
    val_ds = HAM10000Dataset(va_df, cfg, train=False)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg).to(device)
    # n_c: post-split, pre-augmentation counts (Section 3.9)
    criterion = build_loss(cfg, train_ds.class_counts()).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_modules, cfg.weight_decay),
        betas=cfg.adam_betas, eps=cfg.adam_eps)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    key = lambda a, f: (round(a, 4), f)
    best = (-1.0, -1.0)
    best_state = None
    epochs_since_improvement = 0
    rows = []

    for epoch in range(max_epochs):
        # per-group LR schedule (warmup + cosine), preserving the 1:10 ratio
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * cosine_warmup_lambda(
                epoch, cfg.warmup_epochs, max_epochs, base, cfg.min_lr)

        model.train()
        t0, running = time.time(), 0.0
        for x, y, _ in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=cfg.amp and device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)

        val_logits, val_labels, _ = collect_logits(model, val_ld, device)
        vm = evaluate_probs(val_logits, val_labels)
        auprc, f1 = vm["macro_auprc"], vm["macro_f1"]
        rows.append({"epoch": epoch + 1,
                     "train_loss": running / len(train_ds),
                     "val_macro_auprc": auprc, "val_macro_f1": f1,
                     "val_bal_acc": vm["balanced_accuracy"],
                     "lr_backbone": optimizer.param_groups[0]["lr"],
                     "sec": time.time() - t0})
        print(f"  epoch {epoch + 1:02d}/{max_epochs} "
              f"loss={rows[-1]['train_loss']:.4f} "
              f"val AUPRC={auprc:.4f} F1={f1:.2f} "
              f"balacc={vm['balanced_accuracy']:.2f} "
              f"({rows[-1]['sec']:.0f}s)")

        # checkpoint + early-stopping bookkeeping (patience 10, min delta 1e-4)
        if auprc > best[0] + cfg.early_stop_min_delta or \
                (round(auprc, 4) == round(best[0], 4) and f1 > best[1]):
            best = (auprc, f1)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.early_stop_patience:
                print(f"  early stop at epoch {epoch + 1} "
                      f"(no {cfg.early_stop_min_delta} AUPRC gain "
                      f"for {cfg.early_stop_patience} epochs)")
                break

    if log_path:
        pd.DataFrame(rows).to_csv(log_path, index=False)
    if save_best_to and best_state is not None:
        torch.save({"state_dict": best_state,
                    "config": cfg.__dict__,
                    "val_macro_auprc": best[0],
                    "val_macro_f1": best[1]}, save_best_to)
    return best[0], best[1], best_state


# --------------------------------------------------------------------------- #
def run_fold(cfg: Config, fold: int, use_hso: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = os.path.join(cfg.out_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    dev_df, _held_df = outer_fold_frames(cfg.folds_csv, fold)
    tr_df, va_df = inner_split(dev_df, cfg.inner_val_fraction, cfg.split_seed)
    print(f"[fold {fold}] dev={len(dev_df)}  inner-train={len(tr_df)}  "
          f"inner-val={len(va_df)}  (held-out untouched during training)")

    # ------------------------------------------------ nested HSO (optional)
    if use_hso:
        from hso import default_search_space, harmony_search
        proxy_epochs = cfg.hso_epochs if cfg.hso_epochs > 0 else cfg.max_epochs
        if cfg.hso_epochs > 0:
            print(f"[fold {fold}] HSO with {proxy_epochs}-epoch proxy budget "
                  f"— DECLARE THIS IN THE MANUSCRIPT (paper states 60 epochs "
                  f"per candidate).")

        def evaluate(params):
            c = get_config(**{**cfg.__dict__, **params})
            auprc, f1, _ = train_one_config(
                c, tr_df, va_df, device, max_epochs=proxy_epochs)
            return auprc, f1

        best_params, best_score, history = harmony_search(
            evaluate, default_search_space(cfg),
            hms=cfg.hso_hms, hmcr=cfg.hso_hmcr, par=cfg.hso_par,
            bandwidth=cfg.hso_bandwidth, iterations=cfg.hso_iterations,
            seed=cfg.train_seed)
        with open(os.path.join(fold_dir, "hso_history.json"), "w") as f:
            json.dump({"best_params": best_params,
                       "best_score": best_score, "history": history},
                      f, indent=2)
        for k, v in best_params.items():
            setattr(cfg, k, v)
        print(f"[fold {fold}] HSO selected: {best_params}")

    # ------------------------------------------------ final training
    cfg.dump(os.path.join(fold_dir, "config.json"))
    best_auprc, best_f1, _ = train_one_config(
        cfg, tr_df, va_df, device, max_epochs=cfg.max_epochs,
        log_path=os.path.join(fold_dir, "train_log.csv"),
        save_best_to=os.path.join(fold_dir, "best.pt"))
    print(f"[fold {fold}] best inner-val macro AUPRC={best_auprc:.4f} "
          f"macro F1={best_f1:.2f}")

    # ------------------------------------------------ temperature (val only)
    ckpt = torch.load(os.path.join(fold_dir, "best.pt"), map_location=device,
                      weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    val_ld = DataLoader(HAM10000Dataset(va_df, cfg, train=False),
                        batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    val_logits, val_labels, val_ids = collect_logits(model, val_ld, device)
    np.savez(os.path.join(fold_dir, "val_logits.npz"),
             logits=val_logits, labels=val_labels, ids=np.array(val_ids))
    T = fit_temperature(val_logits, val_labels)
    with open(os.path.join(fold_dir, "temperature.json"), "w") as f:
        json.dump({"temperature": T}, f, indent=2)
    print(f"[fold {fold}] fitted temperature T={T:.3f} "
          f"({'sharpening (underconfident model)' if T < 1 else 'smoothing'})")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hso", action="store_true",
                    help="run nested Harmony Search (33 candidates/fold)")
    ap.add_argument("--skip-hso", action="store_true",
                    help="train with reference hyperparameters only")
    ap.add_argument("--hso-epochs", type=int, default=0,
                    help="proxy epoch budget per HSO candidate (0 = full 60)")
    ap.add_argument("--backbone-source", default=None,
                    choices=["timm", "transformers"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="override train_seed in config (default: 42)")
    a = ap.parse_args()

    overrides = {}
    if a.backbone_source:
        overrides["backbone_source"] = a.backbone_source
    if a.out_dir:
        overrides["out_dir"] = a.out_dir
    if a.seed is not None:
        overrides["train_seed"] = a.seed
    overrides["hso_epochs"] = a.hso_epochs
    cfg = get_config(**overrides)

    use_hso = a.hso and not a.skip_hso
    folds = range(cfg.n_folds) if a.all else [a.fold if a.fold is not None else 0]
    for f in folds:
        run_fold(get_config(**overrides), f, use_hso)
