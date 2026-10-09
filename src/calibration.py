"""
calibration.py
Temperature scaling (Sections 3.10 / 4.5).

- Fitted per outer fold on the INNER VALIDATION logits only (never on the
  held-out fold, never on external data) by minimizing NLL over T with L-BFGS.
- Applied to that fold's held-out logits: p = softmax(z / T).
- Argmax is invariant to T, so accuracy is unchanged by construction.
- A fitted T < 1 (the manuscript reports T = 0.155) SHARPENS the distribution,
  i.e. the raw model is UNDERconfident — state this explicitly in Section 4.5.
"""

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray,
                    max_iter: int = 200) -> float:
    z = torch.tensor(val_logits, dtype=torch.float64)
    y = torch.tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = torch.tensor(logits, dtype=torch.float64) / temperature
    return F.softmax(z, dim=1).numpy()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    return apply_temperature(logits, 1.0)
