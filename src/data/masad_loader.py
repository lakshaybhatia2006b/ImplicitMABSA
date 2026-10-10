"""
MASAD Dataset Loader
--------------------
Loads and inspects the MASAD (Multimodal Aspect-Based Sentiment Analysis Dataset).
Dataset: Zhou et al., 2021, Neurocomputing.
Download: https://drive.google.com/file/d/19YJ8vEYCb-uEKUqSGFmysUTvNzxhVKFE/view

Expected raw directory layout after extraction:
  data/raw/
    train.json (or train.tsv / train.txt)
    test.json
    dev.json
    images/  (image files, named by image_id)

This module:
  1. Auto-detects the file format (JSON, TSV, or TXT).
  2. Loads splits into pandas DataFrames.
  3. Validates that required columns are present.
  4. Reports image availability.
  5. Saves a dataset_report.json summary.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "images"
PROCESSED_DIR = Path("data/processed")
REPORT_PATH = Path("outputs/results/dataset_report.json")

SENTIMENT_MAP = {
    "positive": 1, "neutral": 0, "negative": -1,
    "1": 1, "0": 0, "-1": -1,
    "pos": 1, "neu": 0, "neg": -1,
    "2": 1, # If 0=neg, 1=neu, 2=pos
    1: 1, 0: 0, -1: -1, 2: 1
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_split_file(split: str) -> Optional[Path]:
    """Return the first matching file for a given split name."""
    if not RAW_DIR.exists():
        return None
    for ext in [".json", ".tsv", ".csv", ".txt"]:
        direct = RAW_DIR / f"{split}{ext}"
        if direct.exists():
            return direct
        # Check subdirectories
        for found in RAW_DIR.rglob(f"*{split}*{ext}"):
            if found.is_file():
                return found
    return None


def _load_json(path: Path) -> pd.DataFrame:
    """Load JSON-lines or JSON-list format."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        records = json.loads(content)
    else:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    return pd.DataFrame(records)


def _load_tsv(path: Path) -> pd.DataFrame:
    """Load tab-separated format."""
    return pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")


def _load_masad_folder(split: str) -> Optional[pd.DataFrame]:
    """
    Load official MASAD tree structure:
      <split>/text/<sentiment>/<aspect>/<image_id>.txt
      Content format: <image_id>###<title>###<text/description>###<image_url>###<tags>
    """
    records = []
    # Search for directories matching split (e.g. data/raw/train or data/raw/MASAD/train)
    split_candidates = [
        RAW_DIR / split,
        RAW_DIR / "MASAD" / split,
    ]
    for p in RAW_DIR.rglob(split):
        if p.is_dir() and p not in split_candidates:
            split_candidates.append(p)

    split_dir = None
    for cand in split_candidates:
        if (cand / "text").exists():
            split_dir = cand
            break

    if split_dir is None:
        return None

    text_dir = split_dir / "text"
    for txt_file in text_dir.rglob("*.txt"):
        # Expect path like: text_dir / <sentiment> / <aspect> / <image_id>.txt
        rel = txt_file.relative_to(text_dir)
        parts = rel.parts
        if len(parts) >= 3:
            sentiment = parts[0]
            aspect = parts[1]
        elif len(parts) == 2:
            sentiment = parts[0]
            aspect = ""
        else:
            sentiment = ""
            aspect = ""

        try:
            with open(txt_file, encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
        except Exception:
            continue

        chunks = content.split("###")
        image_id = chunks[0].strip() if len(chunks) >= 1 else txt_file.stem
        title = chunks[1].strip() if len(chunks) >= 2 else ""
        desc = chunks[2].strip() if len(chunks) >= 3 else ""
        image_url = chunks[3].strip() if len(chunks) >= 4 else ""
        tags = chunks[4].strip() if len(chunks) >= 5 else ""

        full_text = f"{title} {desc}".strip() or title or desc

        records.append({
            "image_id": image_id,
            "text": full_text,
            "aspect": aspect,
            "sentiment": sentiment,
            "image_url": image_url,
            "tags": tags,
        })

    if records:
        print(f"[INFO] Parsed official MASAD directory structure for '{split}': {len(records)} samples")
        return pd.DataFrame(records)
    return None


def _load_txt(path: Path) -> pd.DataFrame:
    """
    Attempt to parse plain-text ABSA format.
    Expected pattern (one sample per block):
        <image_id>\t<text>
        <aspect>\t<sentiment>
        (blank line)
    Or hash-delimited MASAD single sample line:
        <image_id>###<title>###<text>###<image_url>###<tags>
    """
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip() for l in f.readlines()]

    i = 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        line = lines[i]
        if "###" in line:
            chunks = line.split("###")
            image_id = chunks[0].strip() if len(chunks) >= 1 else ""
            title = chunks[1].strip() if len(chunks) >= 2 else ""
            desc = chunks[2].strip() if len(chunks) >= 3 else ""
            image_url = chunks[3].strip() if len(chunks) >= 4 else ""
            text = f"{title} {desc}".strip() or title or desc
            records.append({
                "image_id": image_id,
                "text": text,
                "aspect": "",
                "sentiment": "",
                "image_url": image_url,
            })
            i += 1
            continue

        parts = line.split("\t")
        if len(parts) >= 2:
            image_id, text = parts[0], "\t".join(parts[1:])
        else:
            image_id, text = "", parts[0]
        i += 1
        # Aspect + sentiment on next line
        if i < len(lines) and lines[i].strip():
            asp_parts = lines[i].split("\t")
            aspect = asp_parts[0] if len(asp_parts) >= 1 else ""
            sentiment = asp_parts[1] if len(asp_parts) >= 2 else ""
            i += 1
        else:
            aspect, sentiment = "", ""
        records.append({
            "image_id": image_id,
            "text": text,
            "aspect": aspect,
            "sentiment": sentiment,
        })
    return pd.DataFrame(records)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names across different dataset variants.
    Maps common synonyms to standard names:
      text, aspect, sentiment, image_id
    """
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    for std, synonyms in {
        "text": ["sentence", "review", "tweet", "content", "caption"],
        "aspect": ["aspect_term", "aspect_category", "category", "target"],
        "sentiment": ["polarity", "label", "sentiment_polarity", "opinion"],
        "image_id": ["image", "img_id", "img", "imageid", "image_path"],
    }.items():
        if std not in col_lower:
            for syn in synonyms:
                if syn in col_lower:
                    rename_map[col_lower[syn]] = std
                    break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _encode_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add an integer label column (1 / 0 / -1)."""
    if "sentiment" in df.columns:
        df["label"] = (
            df["sentiment"].astype(str).str.lower()
            .map(SENTIMENT_MAP)
        )
    return df


def _build_image_index() -> Dict[str, Path]:
    """Index all images in data/raw by their stem/ID."""
    index = {}
    if not RAW_DIR.exists():
        return index
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.gif"]:
        for p in RAW_DIR.rglob(ext):
            if p.is_file():
                index[p.stem] = p
    return index


def _check_images(df: pd.DataFrame) -> Tuple[int, int]:
    """Count how many samples have an accessible image file."""
    if "image_id" not in df.columns or not RAW_DIR.exists():
        return 0, len(df)
    
    img_index = _build_image_index()
    unique_ids = df["image_id"].dropna().astype(str).unique()
    if len(unique_ids) == 0:
        return 0, 0

    found = sum(1 for img_id in unique_ids if img_id in img_index)
    return found, len(unique_ids)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_split(split: str = "train") -> Optional[pd.DataFrame]:
    """
    Load a single split ('train', 'dev', 'test') from data/raw/.
    Returns a normalised DataFrame or None if the file is not found.
    """
    # 1. First check if raw directory contains MASAD folder hierarchy (train/text/...)
    df_folder = _load_masad_folder(split)
    if df_folder is not None and not df_folder.empty:
        df_folder = _normalise_columns(df_folder)
        df_folder = _encode_sentiment(df_folder)
        df_folder["split"] = split
        print(f"  -> {len(df_folder):,} samples, columns: {list(df_folder.columns)}")
        return df_folder

    # 2. Check for individual files (train.json, train.tsv, train.csv, train.txt)
    path = _find_split_file(split)
    if path is None:
        print(f"[WARNING] No file or directory found for split '{split}' in {RAW_DIR}/")
        return None

    print(f"[INFO] Loading {split} from {path}")
    ext = path.suffix.lower()
    if ext == ".json":
        df = _load_json(path)
    elif ext == ".tsv":
        df = _load_tsv(path)
    elif ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    else:
        df = _load_txt(path)

    df = _normalise_columns(df)
    df = _encode_sentiment(df)
    df["split"] = split
    print(f"  -> {len(df):,} samples, columns: {list(df.columns)}")
    return df


def load_all_splits() -> Dict[str, pd.DataFrame]:
    """Load train / dev / test splits and return as a dict."""
    splits = {}
    for name in ["train", "dev", "test"]:
        df = load_split(name)
        if df is not None:
            splits[name] = df
    return splits


def inspect_dataset(splits: Dict[str, pd.DataFrame]) -> dict:
    """
    Print a structured report of the dataset and return it as a dict.
    Saves to outputs/results/dataset_report.json.
    """
    report = {"splits": {}}
    all_dfs = []

    for split, df in splits.items():
        imgs_found, imgs_total = _check_images(df)
        sentiment_dist = {}
        if "label" in df.columns:
            sentiment_dist = df["label"].value_counts().to_dict()
        aspect_sample = []
        if "aspect" in df.columns:
            aspect_sample = df["aspect"].value_counts().head(10).index.tolist()

        info = {
            "n_samples": len(df),
            "columns": list(df.columns),
            "sentiment_distribution": {str(k): int(v) for k, v in sentiment_dist.items()},
            "top_aspects": aspect_sample,
            "images_found": imgs_found,
            "images_total_unique": imgs_total,
            "has_images": IMAGE_DIR.exists() or imgs_found > 0,
        }
        report["splits"][split] = info
        all_dfs.append(df)

        print(f"\n-- {split.upper()} split --")
        print(f"  Samples        : {info['n_samples']:,}")
        print(f"  Columns        : {info['columns']}")
        print(f"  Sentiment dist : {info['sentiment_distribution']}")
        print(f"  Top aspects    : {info['top_aspects']}")
        print(f"  Image dir      : {IMAGE_DIR} (exists={IMAGE_DIR.exists()})")
        print(f"  Images found   : {imgs_found}/{imgs_total} unique IDs")

    if not IMAGE_DIR.exists() and all(s.get("images_found", 0) == 0 for s in report["splits"].values()):
        report["image_availability"] = (
            "LIMITATION: Image directory not found at data/raw/images/. "
            "Text-only pipeline is fully functional. Image-dependent models "
            "require downloading images from the MASAD Google Drive link."
        )
        print("\n[LIMITATION] Images not found -- image-only and fusion models "
              "will use placeholder (zero) tensors for demonstration.")
    else:
        report["image_availability"] = "Images available."

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    splits = load_all_splits()
    if splits:
        inspect_dataset(splits)
    else:
        print("No splits loaded. Please place MASAD data files in data/raw/")
