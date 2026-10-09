"""
dataset.py
HAM10000 dataset, preprocessing and training augmentation (Section 3.3 / Table 4).

Preprocessing (applied to ALL splits, deterministic):
    1. Hair/artifact handling  — morphological black-hat + threshold + inpainting
                                 (DullRazor-style).
    2. Mild color/illumination normalization — Shades-of-Gray color constancy (p=6).
    3. Resize to 224x224 RGB.
    4. ImageNet normalization.

Training augmentation (TRAINING SPLIT ONLY — Table 4 "Augmentation scope"):
    horizontal flip, vertical flip, rotation, zoom (RandomResizedCrop), mild
    color jitter. Never applied to validation or held-out data.

External datasets (ISIC 2017 / PAD-UFES-20 / BCN20000) reuse ExternalDataset
with a CSV of (image_path, class) rows — see external_validation.py.
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASSES, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD, Config
from make_folds import find_image


# --------------------------------------------------------------------------- #
# Deterministic preprocessing (all splits)
# --------------------------------------------------------------------------- #
def remove_hair(img_bgr: np.ndarray, kernel_size: int = 17,
                threshold: int = 10) -> np.ndarray:
    """DullRazor-style hair removal: black-hat -> threshold -> inpaint."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    if mask.sum() == 0:
        return img_bgr
    return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def shades_of_gray(img_bgr: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray color constancy (Finlayson & Trezzi), mild illumination norm."""
    img = img_bgr.astype(np.float32) + 1e-6
    norm = np.power(np.power(img, power).mean(axis=(0, 1)), 1.0 / power)
    norm = norm / (np.sqrt((norm ** 2).sum()) + 1e-6)
    img = img / (norm[None, None, :] * np.sqrt(3.0) + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


def load_preprocessed(path: str, cfg: Config) -> Image.Image:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if cfg.hair_removal:
        img = remove_hair(img)
    if cfg.color_constancy:
        img = shades_of_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(cfg: Config, train: bool) -> transforms.Compose:
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                cfg.img_size, scale=cfg.zoom_scale, ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(cfg.rotation_degrees),
            transforms.ColorJitter(cfg.jitter_brightness, cfg.jitter_contrast,
                                   cfg.jitter_saturation, cfg.jitter_hue),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        norm,
    ])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class HAM10000Dataset(Dataset):
    """Rows of outputs/folds.csv filtered to the requested image_ids."""

    def __init__(self, folds_df: pd.DataFrame, cfg: Config, train: bool):
        self.df = folds_df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.tf = build_transforms(cfg, train)
        self.paths = [find_image(cfg.image_dir, iid) for iid in self.df["image_id"]]
        missing = [p for p in self.paths if not p]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images not found under {cfg.image_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = load_preprocessed(self.paths[i], self.cfg)
        x = self.tf(img)
        y = int(self.df.loc[i, "class_idx"])
        return x, y, self.df.loc[i, "image_id"]

    def class_counts(self) -> np.ndarray:
        """n_c per class (BEFORE augmentation, per Section 3.9)."""
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        for c, n in self.df["class_idx"].value_counts().items():
            counts[int(c)] = int(n)
        return counts


class ExternalDataset(Dataset):
    """CSV with columns: image_path, class  (class in CLASSES). Eval-only."""

    def __init__(self, csv_path: str, cfg: Config):
        self.df = pd.read_csv(csv_path)
        if not {"image_path", "class"} <= set(self.df.columns):
            raise ValueError("External CSV needs columns: image_path, class")
        unknown = set(self.df["class"]) - set(CLASSES)
        if unknown:
            raise ValueError(f"Unknown classes in external CSV: {unknown} "
                             f"— harmonize labels to {CLASSES} first.")
        self.cfg = cfg
        self.tf = build_transforms(cfg, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        img = load_preprocessed(row["image_path"], self.cfg)
        return self.tf(img), CLASS_TO_IDX[row["class"]], os.path.basename(
            row["image_path"])


# --------------------------------------------------------------------------- #
# Split helpers (Section 3.10: 10% lesion-grouped inner validation)
# --------------------------------------------------------------------------- #
def outer_fold_frames(folds_csv: str, fold: int):
    df = pd.read_csv(folds_csv)
    held = df[df["fold"] == fold].reset_index(drop=True)
    dev = df[df["fold"] != fold].reset_index(drop=True)
    return dev, held


def inner_split(dev_df: pd.DataFrame, val_fraction: float, seed: int):
    """Lesion-grouped, class-stratified inner train/val split of the outer
    development data. Group assignment precedes all augmentation."""
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = max(2, round(1.0 / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(dev_df, y=dev_df["class_idx"],
                                     groups=dev_df["lesion_id"]))
    tr = dev_df.iloc[tr_idx].reset_index(drop=True)
    va = dev_df.iloc[va_idx].reset_index(drop=True)
    assert set(tr["lesion_id"]) & set(va["lesion_id"]) == set(), \
        "Inner split leaked lesion IDs"
    return tr, va
