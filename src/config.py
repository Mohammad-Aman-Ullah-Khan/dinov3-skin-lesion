"""
config.py
Central configuration for the DINOv3 ConvNeXt-B + LADA + ADSF + CSGA framework.

Every value here is taken directly from Section 3 / Table 5 of the manuscript:
"DINOv3 ConvNeXt B with Lesion Aware Multiscale Feature Adaptation for
Dermoscopic Skin Lesion Classification".

Values the manuscript does NOT specify are marked  # UNSPECIFIED-IN-PAPER
and must be declared in the manuscript before submission.
"""

import os
from dataclasses import dataclass, field, asdict


# ----------------------------------------------------------------------------- classes
CLASSES = ["MEL", "NV", "BCC", "BKL", "AKIEC", "DF", "VASC"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# HAM10000 metadata 'dx' column -> canonical class name
DX_TO_CLASS = {
    "mel": "MEL", "nv": "NV", "bcc": "BCC", "bkl": "BKL",
    "akiec": "AKIEC", "df": "DF", "vasc": "VASC",
}

# ImageNet normalization (Section 3.3, Table 4)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    data_root: str = "data/HAM10000"                 # contains images/ + HAM10000_metadata.csv
    metadata_csv: str = "data/HAM10000/HAM10000_metadata.csv"
    image_dir: str = "data/HAM10000/images"
    out_dir: str = "outputs"
    folds_csv: str = "outputs/folds.csv"             # produced by make_folds.py

    # ------------------------------------------------------------------ data
    img_size: int = 224                              # Table 5: 224x224 RGB
    n_folds: int = 5                                 # five-fold lesion-grouped CV
    inner_val_fraction: float = 0.10                 # 10% lesion-grouped inner validation
    split_seed: int = 42                             # Table 5
    train_seed: int = 42                             # Table 5
    num_workers: int = 8

    # ------------------------------------------------------------------ backbone
    # "timm"         -> ImageNet-pretrained ConvNeXt-B (fallback / debugging)
    # "transformers" -> DINOv3-pretrained ConvNeXt-B (the configuration used in the paper)
    backbone_source: str = "transformers"
    timm_backbone: str = "convnext_base"
    dinov3_model_id: str = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    # NOTE (reviewer-critical): the manuscript states F2,F3,F4 at 1/4,1/8,1/16, i.e.
    # ConvNeXt stages 1-3, which EXCLUDES the deepest 1024-ch stage and contradicts
    # the 89M-parameter backbone reported in Table 17. The internally consistent
    # configuration is the LAST THREE stages (1/8,1/16,1/32; 256/512/1024 channels),
    # which is the default here. Set feature_stages=(0,1,2) to follow the text
    # literally instead.
    feature_stages: tuple = (1, 2, 3)

    # ------------------------------------------------------------------ modules
    use_lada: bool = True
    use_adsf: bool = True
    use_csga: bool = True
    lada_reduction: int = 8                          # UNSPECIFIED-IN-PAPER (bottleneck MLP ratio)
    lada_morph_channels: int = 64                    # UNSPECIFIED-IN-PAPER (1x1 projection width)
    adsf_dilations: tuple = (1, 3, 5, 7)             # Section 3.7
    # Branch width = channels // adsf_reduction. Eq. 4 read literally at full
    # width adds ~50M params, contradicting Table 17's +3M for ADSF; a shared
    # reduced branch width is the only Table-17-consistent reading. Declare
    # this width in the manuscript.  # UNSPECIFIED-IN-PAPER
    adsf_reduction: int = 4
    adsf_mlp_hidden: int = 256                       # UNSPECIFIED-IN-PAPER
    csga_dim: int = 512                              # common aggregation dim D  # UNSPECIFIED-IN-PAPER
    embed_dim: int = 512                             # Table 5

    # ------------------------------------------------------------------ head / loss (Section 3.9)
    head: str = "cb_adaptive_margin"                 # or "linear_ce" for ablation rows
    loss_beta: float = 0.9999                        # β, fixed, not searched
    loss_m_max: float = 0.35                         # m_max, fixed, not searched
    loss_scale: float = 30.0                         # s, fixed, not searched
    use_class_weights: bool = True                   # ω_c term (ablation: margin-only head sets False)
    use_margin: bool = True

    # ------------------------------------------------------------------ optimization (Table 5)
    lr_backbone: float = 2e-5                        # reference; HSO range [5e-6, 5e-5] log
    lr_modules: float = 2e-4                         # reference; HSO range [5e-5, 5e-4] log
    weight_decay: float = 0.05                       # reference; HSO range [0.01, 0.10]
    dropout: float = 0.20                            # reference; HSO range [0.10, 0.40]
    adam_betas: tuple = (0.90, 0.999)
    adam_eps: float = 1e-8
    warmup_epochs: int = 5                           # linear warmup
    max_epochs: int = 60
    min_lr: float = 1e-6                             # cosine floor
    batch_size: int = 32
    grad_clip: float = 1.0
    amp: bool = True                                 # FP16 autocast
    label_smoothing: float = 0.0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4
    checkpoint_metric: str = "macro_auprc"           # primary; macro_f1 tiebreak

    # ------------------------------------------------------------------ augmentation (Section 3.3)
    rotation_degrees: float = 30.0                   # UNSPECIFIED-IN-PAPER
    zoom_scale: tuple = (0.80, 1.00)                 # UNSPECIFIED-IN-PAPER (RandomResizedCrop scale)
    jitter_brightness: float = 0.10                  # UNSPECIFIED-IN-PAPER ("mild color jitter")
    jitter_contrast: float = 0.10
    jitter_saturation: float = 0.10
    jitter_hue: float = 0.02
    hair_removal: bool = True                        # blackhat + inpainting (Table 4 "artifact handling")
    color_constancy: bool = True                     # Shades-of-Gray, p=6 (Table 4 "mild color norm.")

    # ------------------------------------------------------------------ HSO (Section 3.10 / Table 5)
    hso_enabled: bool = True
    hso_hms: int = 8                                 # harmony memory size
    hso_hmcr: float = 0.90
    hso_par: float = 0.30
    hso_bandwidth: float = 0.05                      # fraction of normalized range
    hso_iterations: int = 25                         # improvisations after HM init
    hso_epochs: int = 0                              # 0 = full 60-epoch candidates (paper-literal,
                                                     # 165 full trainings). Set e.g. 12 for a proxy
                                                     # budget — MUST then be declared in the paper.
    hso_lr_backbone_range: tuple = (5e-6, 5e-5)      # searched in log10 space
    hso_lr_modules_range: tuple = (5e-5, 5e-4)       # searched in log10 space
    hso_wd_range: tuple = (0.01, 0.10)
    hso_dropout_range: tuple = (0.10, 0.40)

    # ------------------------------------------------------------------ calibration / evaluation
    ece_bins: int = 15                               # UNSPECIFIED-IN-PAPER

    def dump(self, path: str):
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg
