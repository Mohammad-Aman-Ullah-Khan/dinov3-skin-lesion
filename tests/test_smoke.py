"""
test_smoke.py
Smoke + protocol-correctness tests.

Run with:
    pytest tests/test_smoke.py -v

Requirements: pip install pytest torch timm pandas scikit-learn
The timm backbone is used (--backbone-source timm) so no DINOv3 checkpoint
is fetched from the internet; architecture and parameter count are identical.
"""

import sys
import os

# Allow importing from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch


def test_model_forward_pass():
    """Build model with timm backbone, run a forward pass, check output shape."""
    from config import get_config, NUM_CLASSES
    from model import build_model

    cfg = get_config(backbone_source="timm")
    model = build_model(cfg).eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, NUM_CLASSES), (
        f"Expected output shape ({batch_size}, {NUM_CLASSES}), got {out.shape}"
    )
    # Logits should be finite
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf"
    print(f"\nSmoke test PASSED: output shape={tuple(out.shape)}, "
          f"logit range=[{out.min():.2f}, {out.max():.2f}]")


def test_loss_forward():
    """Build loss function and run a forward pass."""
    import numpy as np
    from losses import build_loss
    from config import get_config, NUM_CLASSES

    cfg = get_config()
    # HAM10000 approximate class counts
    counts = np.array([1113, 6705, 514, 1099, 327, 115, 142], dtype=np.float64)
    loss_fn = build_loss(cfg, counts)

    logits = torch.randn(8, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (8,))
    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    print(f"\nLoss smoke test PASSED: loss={loss.item():.4f}")


def test_top_k_accuracy():
    """Verify top_k_accuracy returns sensible values."""
    import numpy as np
    from metrics import top_k_accuracy

    # Perfect predictions: argmax == label → top-1 == top-3 == 100%
    n, k = 20, 7
    labels = np.arange(n) % k
    probs = np.zeros((n, k))
    probs[np.arange(n), labels] = 1.0
    assert top_k_accuracy(labels, probs, k=1) == 100.0
    assert top_k_accuracy(labels, probs, k=3) == 100.0

    # Worst case: true label always last → top-1==0%, top-(k)==100%
    probs2 = np.zeros((n, k))
    for i in range(n):
        wrong = (labels[i] + 1) % k
        probs2[i, wrong] = 1.0
    assert top_k_accuracy(labels, probs2, k=1) == 0.0
    assert top_k_accuracy(labels, probs2, k=k) == 100.0
    print("\ntop_k_accuracy smoke test PASSED")


def test_temperature_argmax_invariance():
    """Temperature scaling must never change the argmax prediction.
    Codifies the fix in test.py: raw vs calibrated argmax must agree 100%.
    """
    from calibration import apply_temperature, softmax_np

    np.random.seed(0)
    logits = np.random.randn(200, 7).astype(np.float32) * 5.0

    for T in (0.155, 0.5, 1.0, 2.0, 5.0):   # covers both sharpening and smoothing
        probs_raw = softmax_np(logits)
        probs_cal = apply_temperature(logits, T)
        agree = float((probs_raw.argmax(1) == probs_cal.argmax(1)).mean() * 100)
        assert agree == 100.0, (
            f"T={T}: argmax agreement is {agree:.2f}% — temperature scaling "
            "must not change any prediction"
        )
    print("\ntemperature argmax invariance test PASSED (T ∈ {0.155, 0.5, 1.0, 2.0, 5.0})")


def test_inner_split_no_lesion_leak():
    """Inner train/val split must not share any lesion_id between partitions.
    This is the single most critical data-leakage guarantee in the protocol.
    A failure here would invalidate all reported metrics.
    """
    import pandas as pd
    from dataset import inner_split

    # Synthetic dev set: 200 images, 50 lesions (4 images each), 7 classes
    np.random.seed(42)
    n_lesions = 50
    n_classes = 7
    lesion_ids = [f"HAM_{i:05d}" for i in range(n_lesions)]
    # each lesion has 4 images, class assigned per lesion
    rows = []
    for lid in lesion_ids:
        cls = np.random.randint(0, n_classes)
        for _ in range(4):
            rows.append({"lesion_id": lid, "class_idx": cls})
    dev_df = pd.DataFrame(rows)

    tr, va = inner_split(dev_df, val_fraction=0.1, seed=42)

    overlap = set(tr["lesion_id"]) & set(va["lesion_id"])
    assert overlap == set(), (
        f"Inner split leaked lesion IDs — this is a critical protocol violation: "
        f"{overlap}"
    )
    assert len(tr) + len(va) == len(dev_df), "Inner split lost rows"
    print(f"\nlesion-leak test PASSED: train={len(tr)}, val={len(va)}, "
          f"overlap={len(overlap)}")
