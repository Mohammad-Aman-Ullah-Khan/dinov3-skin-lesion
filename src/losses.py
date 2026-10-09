"""
losses.py
Class-Balanced Adaptive-Margin objective (Section 3.9, Eqs. 8-9).

Given the scaled cosine logits a_{j,c} = s * q_{j,c} produced by the
CosineClassifier:

    omega_c = (1 - beta) / (1 - beta^{n_c})     (effective-number weight,
              then normalized so mean weight = 1)                Eq. 8
    m_c     = m_max * (n_min / n_c)^{1/4}       (larger margin for rarer
              classes — LDAM-style exponent)                     Eq. 8
    L       = -(1/N) * sum_j omega_{y_j} * log p^m_{j,y_j}       Eq. 9
    where the margin s*m_{y_j} is subtracted from the TARGET logit only.

Class counts n_c are computed on the training partition AFTER lesion-grouped
splitting and BEFORE augmentation (Section 3.9). Margins and weights are used
only during training; inference uses the raw scaled logits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CLASSES


class ClassBalancedAdaptiveMarginLoss(nn.Module):
    def __init__(self, class_counts: np.ndarray, beta: float = 0.9999,
                 m_max: float = 0.35, scale: float = 30.0,
                 use_class_weights: bool = True, use_margin: bool = True):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        missing_cls = [CLASSES[i] for i in np.where(counts == 0)[0]]
        assert (counts > 0).all(), (
            f"Missing classes in training split: {missing_cls}. "
            "This can happen with StratifiedGroupKFold on very rare classes. "
            "Check make_folds.py output for per-fold class counts."
        )
        k = len(counts)

        # ---- effective-number class weights, normalized to mean 1 (Eq. 8)
        eff = (1.0 - beta) / (1.0 - np.power(beta, counts))
        omega = k * eff / eff.sum()

        # ---- distribution-aware margins (Eq. 8)
        n_min = counts.min()
        margins = m_max * np.power(n_min / counts, 0.25)

        self.register_buffer("omega", torch.tensor(
            omega if use_class_weights else np.ones(k), dtype=torch.float32))
        self.register_buffer("margins", torch.tensor(
            margins if use_margin else np.zeros(k), dtype=torch.float32))
        self.scale = float(scale)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits are already scaled cosine scores a = s*q; subtract s*m_y from
        # the target logit only (Eq. 9).
        m_y = self.margins[target]                          # (N,)
        adjusted = logits.clone()
        adjusted.scatter_add_(
            1, target.unsqueeze(1), (-self.scale * m_y).unsqueeze(1))
        log_p = F.log_softmax(adjusted, dim=1)
        nll = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        return (self.omega[target] * nll).mean()


def build_loss(cfg, class_counts: np.ndarray) -> nn.Module:
    if cfg.head == "cb_adaptive_margin":
        return ClassBalancedAdaptiveMarginLoss(
            class_counts, beta=cfg.loss_beta, m_max=cfg.loss_m_max,
            scale=cfg.loss_scale, use_class_weights=cfg.use_class_weights,
            use_margin=cfg.use_margin)
    # Ablation rows with a plain linear head use ordinary cross-entropy.
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


if __name__ == "__main__":
    # Sanity: HAM10000 counts -> margins/weights must be monotone in rarity.
    counts = np.array([1113, 6705, 514, 1099, 327, 115, 142])
    loss = ClassBalancedAdaptiveMarginLoss(counts)
    print("omega  :", loss.omega.numpy().round(3))    # largest for DF
    print("margins:", loss.margins.numpy().round(3))  # m_max=0.35 at DF (n_min)
    logits = torch.randn(8, 7) * 30
    y = torch.randint(0, 7, (8,))
    print("loss   :", float(loss(logits, y)))
