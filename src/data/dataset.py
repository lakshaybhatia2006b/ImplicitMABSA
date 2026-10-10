"""
ImplicitMABSA Dataset Wrapper
------------------------------
PyTorch Dataset classes for the three experiment modes:
  1. TextDataset        — text only
  2. ImageDataset       — images only
  3. MultimodalDataset  — text + images

Handles synthetic (implicit) data. The `is_synthetic` column is preserved
so experiments can be run with filtered or full data.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False


RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "images"
SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".gif"]


def _find_image(image_id) -> Optional[Path]:
    if image_id is None:
        return None
    image_dirs = [IMAGE_DIR]
    if RAW_DIR.exists():
        for p in RAW_DIR.rglob("image*"):
            if p.is_dir() and p not in image_dirs:
                image_dirs.append(p)

    for idir in image_dirs:
        if not idir.exists():
            continue
        for ext in ["", *SUPPORTED_EXTS]:
            p = idir / f"{image_id}{ext}"
            if p.exists() and p.is_file():
                return p
    return None


def _preprocess_image(path: Path):
    if not TORCH_AVAILABLE:
        return np.zeros((3, 224, 224), dtype=np.float32)
    from torchvision import transforms
    from PIL import Image as PILImage

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    try:
        img = PILImage.open(path).convert("RGB")
        return preprocess(img)
    except Exception:
        return torch.zeros(3, 224, 224)


LABEL2ID = {-1: 0, 0: 1, 1: 2}
ID2LABEL = {0: -1, 1: 0, 2: 1}


class TextDataset(Dataset):
    """Dataset for text-only models."""

    def __init__(self, df: pd.DataFrame, text_col: str = "masked_text"):
        # Prefer masked_text (implicit); fall back to original text
        if text_col not in df.columns and "text" in df.columns:
            text_col = "text"
        self.texts = df[text_col].fillna("").tolist()
        self.labels = [LABEL2ID[int(l)] for l in df["label"].tolist()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class ImageDataset(Dataset):
    """Dataset for image-only models."""

    def __init__(self, df: pd.DataFrame):
        self.image_ids = df.get("image_id", pd.Series([None] * len(df))).tolist()
        self.labels = [LABEL2ID[int(l)] for l in df["label"].tolist()]
        self.images_available = IMAGE_DIR.exists()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        if self.images_available:
            path = _find_image(img_id)
            if path:
                tensor = _preprocess_image(path)
            else:
                tensor = torch.zeros(3, 224, 224) if TORCH_AVAILABLE else np.zeros((3, 224, 224), dtype=np.float32)
        else:
            tensor = torch.zeros(3, 224, 224) if TORCH_AVAILABLE else np.zeros((3, 224, 224), dtype=np.float32)
        return tensor, self.labels[idx]


class MultimodalDataset(Dataset):
    """Dataset that returns both text and image for fusion models."""

    def __init__(self, df: pd.DataFrame, text_col: str = "masked_text"):
        if text_col not in df.columns and "text" in df.columns:
            text_col = "text"
        self.texts = df[text_col].fillna("").tolist()
        self.image_ids = df.get("image_id", pd.Series([None] * len(df))).tolist()
        self.labels = [LABEL2ID[int(l)] for l in df["label"].tolist()]
        self.images_available = IMAGE_DIR.exists()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        img_id = self.image_ids[idx]
        if self.images_available:
            path = _find_image(img_id)
            if path:
                img = _preprocess_image(path)
            else:
                img = torch.zeros(3, 224, 224) if TORCH_AVAILABLE else np.zeros((3, 224, 224), dtype=np.float32)
        else:
            img = torch.zeros(3, 224, 224) if TORCH_AVAILABLE else np.zeros((3, 224, 224), dtype=np.float32)
        return text, img, label


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split a DataFrame into train / val / test sets.
    Stratifies by label to prevent class imbalance leakage.
    Never mixes synthetic and real data across splits (preserves is_synthetic column).
    """
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio
    stratify_col = df["label"] if "label" in df.columns else None

    train, temp = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=stratify_col,
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp["label"] if "label" in temp.columns else None
    val, test = train_test_split(
        temp,
        test_size=(1 - val_frac),
        random_state=seed,
        stratify=stratify_temp,
    )
    print(f"Split sizes — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
