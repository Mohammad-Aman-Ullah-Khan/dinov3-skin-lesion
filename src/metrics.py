"""
metrics.py
Every metric reported in Tables 7-12 and Section 4:

  accuracy, balanced accuracy, macro precision/recall/F1, multiclass MCC,
  Cohen's kappa, macro AUROC (OvR), macro AUPRC (OvR), per-class OvR
  sensitivity/specificity/precision/NPV/F1, NLL, multiclass Brier score,
  expected calibration error (ECE, equal-width confidence bins).
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from config import CLASSES, NUM_CLASSES


def _one_hot(y: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    oh = np.zeros((len(y), k))
    oh[np.arange(len(y)), y] = 1.0
    return oh


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    """Top-k accuracy: fraction of samples where true label is in top-k predictions."""
    top_k = np.argsort(probs, axis=1)[:, -k:]          # (N, k) highest-prob indices
    correct = np.any(top_k == y_true[:, None], axis=1)  # True if true label in top-k
    return float(correct.mean() * 100)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict:
    """Aggregate metrics (Tables 7-8). probs: (N, 7) softmax probabilities."""
    y_pred = probs.argmax(axis=1)
    oh = _one_hot(y_true)
    out = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "macro_precision": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0) * 100,
        "macro_recall": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             zero_division=0) * 100,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        out["macro_auroc"] = roc_auc_score(oh, probs, average="macro")
        out["macro_auprc"] = average_precision_score(oh, probs, average="macro")
    except ValueError:  # a class absent from y_true (small external subsets)
        out["macro_auroc"] = float("nan")
        out["macro_auprc"] = float("nan")
    return out


def per_class_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict:
    """One-vs-rest per-class table (Table 9) + per-class AUROC/AUPRC (Table 10)."""
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    rows = {}
    for c, name in enumerate(CLASSES):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        binary = (y_true == c).astype(int)
        rows[name] = {
            "n": int(cm[c].sum()),
            "sensitivity": 100 * tp / max(tp + fn, 1),
            "specificity": 100 * tn / max(tn + fp, 1),
            "precision": 100 * tp / max(tp + fp, 1),
            "npv": 100 * tn / max(tn + fn, 1),
            "f1": 100 * 2 * tp / max(2 * tp + fp + fn, 1),
            "auroc": roc_auc_score(binary, probs[:, c])
                     if 0 < binary.sum() < len(binary) else float("nan"),
            "auprc": average_precision_score(binary, probs[:, c])
                     if binary.sum() > 0 else float("nan"),
            "prevalence": 100 * binary.mean(),
        }
    return rows


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray,
                        n_bins: int = 15) -> Dict:
    """NLL, multiclass Brier score, top-label ECE (Table 12)."""
    eps = 1e-12
    p_true = probs[np.arange(len(y_true)), y_true]
    nll = float(-np.log(np.clip(p_true, eps, 1.0)).mean())
    brier = float(((probs - _one_hot(y_true)) ** 2).sum(axis=1).mean())

    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return {"nll": nll, "brier": brier, "ece_percent": 100 * ece}


def error_table(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Class-specific error analysis (Table 11)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    rows = {}
    for c, name in enumerate(CLASSES):
        n = int(cm[c].sum())
        correct = int(cm[c, c])
        errors = n - correct
        off = cm[c].copy()
        off[c] = 0
        top = off.argmax()
        rows[name] = {
            "n": n, "correct": correct, "errors": errors,
            "error_rate": 100 * errors / max(n, 1),
            "most_frequent_misclassification":
                f"{CLASSES[top]} ({off[top]})" if errors else "—",
        }
    rows["Overall"] = {
        "n": int(cm.sum()), "correct": int(np.trace(cm)),
        "errors": int(cm.sum() - np.trace(cm)),
        "error_rate": 100 * (cm.sum() - np.trace(cm)) / cm.sum(),
        "most_frequent_misclassification": "—",
    }
    return rows
