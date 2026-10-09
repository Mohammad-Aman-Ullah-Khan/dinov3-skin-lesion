"""
model.py
Proposed architecture (Sections 3.4-3.9, Figure 1):

    RGB 224x224
      -> DINOv3-pretrained ConvNeXt-B backbone -> hierarchical features F2, F3, F4
      -> LADA  (Eq. 1-3): appearance channel attention + morphology spatial
               attention, concat -> 1x1 fusion -> residual
      -> ADSF  (Eq. 4-5): 1x1 local + dilated 3x3 (r=1,3,5,7) + global context,
               input-dependent softmax scale weights, 1x1 fusion
      -> CSGA  (Eq. 6): align to common D x Ha x Wa, per-level sigmoid channel
               gates, gated sum
      -> GAP -> phi -> 512-D embedding (Eq. 7)
      -> normalized cosine classifier, scale s (margins applied only in the loss,
         never at inference — Section 3.9).

Backbone note (reviewer-critical): the manuscript text says F2/F3/F4 sit at
1/4, 1/8, 1/16 of input resolution (ConvNeXt stages 1-3), but Table 17's 89M /
15.5 GFLOP backbone is the FULL ConvNeXt-B, which requires the deepest stage.
Default here: last three stages (1/8, 1/16, 1/32 — channels 256/512/1024),
selected with cfg.feature_stages=(1,2,3). Use (0,1,2) to follow the text
literally.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, NUM_CLASSES


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
class ConvNeXtBackbone(nn.Module):
    """Wraps either a timm ConvNeXt-B (ImageNet) or the DINOv3 ConvNeXt-B from
    HuggingFace transformers, returning the three requested stage outputs."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.stages: Sequence[int] = tuple(cfg.feature_stages)
        self.source = cfg.backbone_source

        if self.source == "timm":
            import timm
            self.net = timm.create_model(
                cfg.timm_backbone, pretrained=True, features_only=True,
                out_indices=self.stages)
            self.channels = list(self.net.feature_info.channels())
        elif self.source == "transformers":
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(cfg.dinov3_model_id)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size)
                feats = self._forward_transformers(dummy)
            self.channels = [f.shape[1] for f in feats]
        else:
            raise ValueError(f"Unknown backbone_source: {self.source}")

    def _forward_transformers(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.net(pixel_values=x, output_hidden_states=True)
        # hidden_states = (embeddings, stage1, stage2, stage3, stage4) NCHW
        stage_maps = list(out.hidden_states[1:])
        return [stage_maps[i] for i in self.stages]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.source == "timm":
            return list(self.net(x))
        return self._forward_transformers(x)


# --------------------------------------------------------------------------- #
# LADA — Lesion Aware Dual Attention (Section 3.6, Eqs. 1-3, Figure 2)
# --------------------------------------------------------------------------- #
class LADA(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, morph_channels: int = 64):
        super().__init__()
        hidden = max(channels // reduction, 8)
        # Appearance branch: [mu || sigma] (2C) -> bottleneck MLP -> sigmoid (Eq. 1)
        self.app_mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        # Morphology branch (Eq. 2): 1x1 proj -> DW 3x3 || DW 7x7 -> 1x1 -> sigmoid
        self.morph_proj = nn.Conv2d(channels, morph_channels, 1)
        self.dw3 = nn.Conv2d(morph_channels, morph_channels, 3, padding=1,
                             groups=morph_channels)
        self.dw7 = nn.Conv2d(morph_channels, morph_channels, 7, padding=3,
                             groups=morph_channels)
        self.morph_out = nn.Conv2d(2 * morph_channels, 1, 1)
        # Fusion (Eq. 3): concat(F_app, F_morph) 2C -> 1x1 -> C, residual add
        self.fuse = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # ---- appearance channel attention (Eq. 1)
        mu = x.mean(dim=(2, 3))                                     # (B, C)
        sigma = torch.sqrt(x.var(dim=(2, 3), unbiased=False) + 1e-6)
        a_app = torch.sigmoid(self.app_mlp(torch.cat([mu, sigma], dim=1)))
        f_app = x * a_app.view(b, c, 1, 1)
        # ---- morphology spatial attention (Eq. 2)
        p = self.morph_proj(x)
        a_morph = torch.sigmoid(
            self.morph_out(torch.cat([self.dw3(p), self.dw7(p)], dim=1)))
        f_morph = x * a_morph                                       # broadcast over C
        # ---- fusion + residual (Eq. 3)
        return x + self.fuse(torch.cat([f_app, f_morph], dim=1))


# --------------------------------------------------------------------------- #
# ADSF — Adaptive Dilated Scale Fusion (Section 3.7, Eqs. 4-5, Figure 3)
# --------------------------------------------------------------------------- #
class ADSF(nn.Module):
    """All six branches share the SAME reduced channel width b = C // reduction
    ("all six branches return the same channel dimensionality", Section 3.7);
    the final 1x1 fusion restores C. A full-width reading of Eq. 4 adds ~50M
    parameters and contradicts Table 17 (+3M for ADSF)."""

    def __init__(self, channels: int, dilations=(1, 3, 5, 7),
                 mlp_hidden: int = 256, reduction: int = 4):
        super().__init__()
        b = max(channels // reduction, 32)
        self.reduce = nn.Conv2d(channels, b, 1)                      # shared width
        self.local = nn.Conv2d(b, b, 1)                              # B_{i,0}
        self.dilated = nn.ModuleList([
            nn.Conv2d(b, b, 3, padding=r, dilation=r)                # B_{i,r}
            for r in dilations
        ])
        self.global_proj = nn.Conv2d(b, b, 1)                        # B_{i,g}
        n_branches = 2 + len(dilations)
        self.weight_mlp = nn.Sequential(                             # Eq. 5
            nn.Linear(n_branches * b, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, n_branches),
        )
        self.fuse = nn.Conv2d(b, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xr = self.reduce(x)
        branches = [self.local(xr)]
        branches += [conv(xr) for conv in self.dilated]
        g = self.global_proj(F.adaptive_avg_pool2d(xr, 1))
        branches.append(g.expand(-1, -1, h, w))                      # Up_{HxW}
        # input-dependent softmax scale weights (Eq. 5)
        z = torch.cat([bk.mean(dim=(2, 3)) for bk in branches], dim=1)
        alpha = torch.softmax(self.weight_mlp(z), dim=1)             # (B, K)
        fused = sum(alpha[:, k].view(b, 1, 1, 1) * bk
                    for k, bk in enumerate(branches))
        return self.fuse(fused)


# --------------------------------------------------------------------------- #
# CSGA — Cross Scale Gated Aggregation (Section 3.8, Eq. 6)
# --------------------------------------------------------------------------- #
class CSGA(nn.Module):
    """Aligns the three refined levels to D x Ha x Wa (Ha,Wa = deepest level's
    grid), gates each level with a sigmoid channel gate from its GAP
    descriptor, and sums."""

    def __init__(self, in_channels: List[int], dim: int = 512):
        super().__init__()
        self.align = nn.ModuleList(
            [nn.Conv2d(c, dim, 1) for c in in_channels])             # Align_i
        self.gates = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in in_channels])              # W_{g,i}, b_{g,i}
        self.dim = dim

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        target_hw = feats[-1].shape[-2:]
        out = 0.0
        for f, align, gate in zip(feats, self.align, self.gates):
            fa = align(f)
            if fa.shape[-2:] != target_hw:
                fa = F.adaptive_avg_pool2d(fa, target_hw)
            d = fa.mean(dim=(2, 3))                                  # GAP descriptor
            g = torch.sigmoid(gate(d)).unsqueeze(-1).unsqueeze(-1)   # (B, D, 1, 1)
            out = out + g * fa                                       # Eq. 6 sum
        return out


# --------------------------------------------------------------------------- #
# Cosine classifier (margins live in the loss, not here — Section 3.9)
# --------------------------------------------------------------------------- #
class CosineClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        # a_{j,c} = s * cos(theta) (Eq. 8); inference uses these logits directly.
        return self.scale * F.linear(F.normalize(e, dim=1),
                                     F.normalize(self.weight, dim=1))


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class LesionAdaptiveNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = ConvNeXtBackbone(cfg)
        chans = self.backbone.channels

        self.lada = nn.ModuleList(
            [LADA(c, cfg.lada_reduction, cfg.lada_morph_channels) for c in chans]
        ) if cfg.use_lada else None
        self.adsf = nn.ModuleList(
            [ADSF(c, cfg.adsf_dilations, cfg.adsf_mlp_hidden,
                  cfg.adsf_reduction) for c in chans]
        ) if cfg.use_adsf else None

        if cfg.use_csga:
            self.csga = CSGA(chans, cfg.csga_dim)
            feat_dim = cfg.csga_dim
        else:
            # Ablation rows without CSGA: pool the deepest level only.
            self.csga = None
            feat_dim = chans[-1]

        # Eq. 7: phi(GAP(F_CSGA)) -> 512-D embedding
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, cfg.embed_dim),
            nn.BatchNorm1d(cfg.embed_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(cfg.dropout)

        if cfg.head == "cb_adaptive_margin":
            self.classifier = CosineClassifier(cfg.embed_dim, NUM_CLASSES,
                                               cfg.loss_scale)
        else:  # plain linear head for backbone/+LADA/+ADSF/+CSGA ablation rows
            self.classifier = nn.Linear(cfg.embed_dim, NUM_CLASSES)

    # -- parameter groups: backbone lr vs. new-module lr (Section 3.10) -------
    def param_groups(self, lr_backbone: float, lr_modules: float,
                     weight_decay: float):
        no_decay_kw = ("bias", "bn", "norm")
        groups = {("bb", True): [], ("bb", False): [], ("new", True): [],
                  ("new", False): []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_bb = name.startswith("backbone.")
            decay = not any(k in name.lower() for k in no_decay_kw) and p.ndim > 1
            groups[("bb" if is_bb else "new", decay)].append(p)
        return [
            {"params": groups[("bb", True)], "lr": lr_backbone,
             "weight_decay": weight_decay},
            {"params": groups[("bb", False)], "lr": lr_backbone,
             "weight_decay": 0.0},
            {"params": groups[("new", True)], "lr": lr_modules,
             "weight_decay": weight_decay},
            {"params": groups[("new", False)], "lr": lr_modules,
             "weight_decay": 0.0},
        ]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.lada is not None:
            feats = [m(f) for m, f in zip(self.lada, feats)]
        if self.adsf is not None:
            feats = [m(f) for m, f in zip(self.adsf, feats)]
        if self.csga is not None:
            agg = self.csga(feats)
        else:
            agg = feats[-1]
        pooled = agg.mean(dim=(2, 3))                                # GAP
        return self.embed(pooled)                                    # 512-D e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.dropout(self.forward_features(x))
        return self.classifier(e)


def build_model(cfg: Config) -> LesionAdaptiveNet:
    return LesionAdaptiveNet(cfg)


if __name__ == "__main__":
    # Shape smoke test with the timm fallback (no DINOv3 download needed).
    from config import get_config
    cfg = get_config(backbone_source="timm", dropout=0.2)
    m = build_model(cfg)
    x = torch.randn(2, 3, 224, 224)
    logits = m(x)
    print("channels:", m.backbone.channels)
    print("logits:", tuple(logits.shape))
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"params: {n:.1f}M")
