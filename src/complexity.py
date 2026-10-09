"""
complexity.py
Computational complexity and inference efficiency (Section 4.9 / Table 17):
parameters (M), GFLOPs, model size (MB), peak GPU memory (GB), GPU latency
(ms/image, batch 1), CPU latency (ms/image), throughput (images/s), for the
incremental configurations and for FP32 vs FP16.

GFLOPs via fvcore (pip install fvcore); falls back to thop; else params only.

Usage:
    python complexity.py                 # all incremental rows, FP32 + FP16
    python complexity.py --rows full
"""

import argparse
import time

import numpy as np
import torch

from ablation import ROWS
from config import get_config
from model import build_model


def count_gflops(model, x):
    try:
        from fvcore.nn import FlopCountAnalysis
        return FlopCountAnalysis(model, x).total() / 1e9
    except Exception:
        try:
            from thop import profile
            flops, _ = profile(model, inputs=(x,), verbose=False)
            return flops / 1e9
        except Exception:
            return float("nan")


@torch.no_grad()
def latency(model, device, dtype, n_warmup=10, n_runs=50, batch=1):
    x = torch.randn(batch, 3, 224, 224, device=device, dtype=dtype)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_runs / batch * 1000
    return ms


@torch.no_grad()
def profile_row(name: str, overrides: dict, do_cpu: bool):
    print(f"\n=== {name} ===")
    cfg = get_config(**overrides)
    model = build_model(cfg).eval()
    x = torch.randn(1, 3, 224, 224)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = count_gflops(model, x)
    size_fp32 = sum(p.numel() * 4 for p in model.parameters()) / 1e6
    print(f"Parameters: {params_m:.1f} M | GFLOPs: {gflops:.1f} | "
          f"FP32 size: {size_fp32:.0f} MB | FP16 size: {size_fp32 / 2:.0f} MB")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        for dtype, tag in ((torch.float32, "FP32"), (torch.float16, "FP16")):
            m = build_model(cfg).eval().to(dev, dtype=dtype)
            torch.cuda.reset_peak_memory_stats()
            ms = latency(m, dev, dtype, batch=1)
            ms_batch32 = latency(m, dev, dtype, batch=32)
            thr = 1000.0 / ms_batch32
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{tag}: GPU latency {ms:.1f} ms/img | "
                  f"throughput {thr:.1f} img/s (batch 32) | "
                  f"peak GPU mem {peak:.2f} GB")
            del m
            torch.cuda.empty_cache()
    else:
        print("CUDA unavailable — GPU rows will be N.M.")

    if do_cpu:
        ms = latency(model, torch.device("cpu"), torch.float32,
                     n_warmup=2, n_runs=10)
        print(f"CPU latency: {ms:.0f} ms/img (FP32)")
    else:
        print("CPU latency: N.M. (pass --cpu to measure)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=list(ROWS))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--backbone-source", default="timm",
                    choices=["timm", "transformers"],
                    help="timm avoids the DINOv3 download; architecture and "
                         "therefore params/GFLOPs are identical")
    a = ap.parse_args()
    for row in a.rows:
        overrides = dict(ROWS[row])
        overrides["backbone_source"] = a.backbone_source
        profile_row(row, overrides, a.cpu)
