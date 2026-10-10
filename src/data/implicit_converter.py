"""
Implicit Aspect Conversion Pipeline
------------------------------------
Converts MASAD explicit-ABSA samples into candidate implicit-ABSA samples
by masking (removing) explicit aspect terms from the text.

Principles:
  - Original data is NEVER modified.
  - Every transformed sample is marked: is_synthetic=True.
  - A confidence score (0-1) reflects how reliably the sentiment is still
    inferable after masking.
  - A manual_validation_flag marks samples recommended for human review.
  - Transformation logic is documented; no results are misrepresented as
    naturally-occurring implicit data.

Output:
  data/synthetic/implicit_<split>.json
  data/synthetic/implicit_conversion_log.json
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
SYNTHETIC_DIR = Path("data/synthetic")
LOG_PATH = SYNTHETIC_DIR / "implicit_conversion_log.json"

# Minimum word-count of the masked text to retain the sample
MIN_WORDS_AFTER_MASK = 4

# Sentiment words that serve as contextual clues (raises confidence)
_POS_WORDS = {"great", "excellent", "amazing", "love", "loved", "beautiful",
              "awesome", "fantastic", "good", "nice", "best", "perfect",
              "wonderful", "delicious", "outstanding", "superb", "happy"}
_NEG_WORDS = {"bad", "terrible", "horrible", "worst", "hate", "hated",
              "awful", "poor", "disappointing", "mediocre", "slow",
              "expensive", "broken", "fail", "failed", "wrong", "ugly"}
_NEU_WORDS = {"okay", "ok", "average", "normal", "typical", "usual",
              "standard", "decent", "moderate", "fine"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalize unicode and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _mask_aspect(text: str, aspect: str) -> Optional[str]:
    """
    Remove all case-insensitive occurrences of the aspect term (and its plural) from text.
    Uses word boundaries to avoid creating grammatical fragments like 'bad s'.
    Returns masked text or None if the text collapses too much.
    """
    if not aspect or not text:
        return None
    cleaned_aspect = aspect.strip()
    # Match singular and common plural forms with word boundary awareness
    plural_aspect = cleaned_aspect + "s" if not cleaned_aspect.endswith("s") else cleaned_aspect
    pattern = re.compile(rf"\b({re.escape(plural_aspect)}|{re.escape(cleaned_aspect)})\b", re.IGNORECASE)
    masked = pattern.sub("", text)
    masked = _clean(masked)
    if len(masked.split()) < MIN_WORDS_AFTER_MASK:
        return None
    return masked


def _sentiment_confidence(masked_text: str, label: int) -> float:
    """
    Heuristic confidence that the original sentiment label is still inferable
    from the masked text.
    Returns a float 0.0 – 1.0.
    """
    tokens = set(masked_text.lower().split())
    if label == 1:
        clue_count = len(tokens & _POS_WORDS)
    elif label == -1:
        clue_count = len(tokens & _NEG_WORDS)
    else:
        clue_count = len(tokens & _NEU_WORDS)

    # Base confidence from clue words; cap at 0.95
    base = min(0.4 + 0.2 * clue_count, 0.95)
    return round(base, 2)


def _needs_manual_review(confidence: float, label: int,
                         masked_text: str) -> bool:
    """Flag samples that a human annotator should verify."""
    if confidence < 0.5:
        return True
    if label == 0:  # neutral is hardest to infer
        return True
    if len(masked_text.split()) < 7:
        return True
    return False


# ── Core Conversion ────────────────────────────────────────────────────────────

def convert_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Convert one split of explicit ABSA data to candidate implicit samples.

    Returns a new DataFrame with columns:
      original_text, masked_text, aspect, label, sentiment,
      image_id, is_synthetic, confidence, manual_validation_flag,
      conversion_note, split
    """
    if "text" not in df.columns or "aspect" not in df.columns:
        raise ValueError("DataFrame must have 'text' and 'aspect' columns.")

    rows: List[dict] = []
    stats = {"total": len(df), "converted": 0, "skipped": 0,
             "skip_reasons": {}}

    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        aspect = str(row.get("aspect", ""))
        label = row.get("label", None)
        sentiment = str(row.get("sentiment", ""))
        image_id = row.get("image_id", None)

        # Skip if essential fields are missing
        if not text or not aspect or pd.isna(label):
            stats["skipped"] += 1
            reason = "missing_field"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            continue

        # Perform masking
        masked = _mask_aspect(text, aspect)
        if masked is None:
            stats["skipped"] += 1
            reason = "text_too_short_after_mask"
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            continue

        confidence = _sentiment_confidence(masked, int(label))
        flag = _needs_manual_review(confidence, int(label), masked)

        rows.append({
            "original_text": text,
            "masked_text": masked,
            "aspect": aspect,           # preserved for reference / evaluation
            "label": int(label),
            "sentiment": sentiment,
            "image_id": image_id,
            "is_synthetic": True,       # ← explicit provenance marker
            "source": "MASAD_explicit_masked",
            "confidence": confidence,
            "manual_validation_flag": flag,
            "conversion_note": (
                f"Aspect '{aspect}' removed from explicit ABSA sample. "
                "Label retained only if contextual clues sufficient. "
                "NOT a naturally-occurring implicit sample."
            ),
            "split": split,
        })
        stats["converted"] += 1

    synthetic_df = pd.DataFrame(rows)
    print(f"[{split}] Converted: {stats['converted']:,} / "
          f"{stats['total']:,} total | Skipped: {stats['skipped']:,}")
    print(f"  Skip reasons: {stats['skip_reasons']}")

    # Save to disk
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / f"implicit_{split}.json"
    synthetic_df.to_json(out_path, orient="records", indent=2, force_ascii=False)
    print(f"  Saved to {out_path}")

    stats["split"] = split
    return synthetic_df, stats


def run_pipeline(splits: dict) -> dict:
    """
    Convert all splits and write a summary conversion log.
    
    Parameters
    ----------
    splits : dict[str, pd.DataFrame]  from masad_loader.load_all_splits()

    Returns
    -------
    dict of converted DataFrames keyed by split name.
    """
    results = {}
    all_stats = []

    for split_name, df in splits.items():
        syn_df, stats = convert_split(df, split_name)
        results[split_name] = syn_df
        all_stats.append(stats)

    # Save log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump({
            "description": (
                "Implicit conversion log. All synthetic samples are derived from "
                "MASAD explicit ABSA data by masking aspect terms. "
                "is_synthetic=True on every row. These are NOT naturally-occurring "
                "implicit examples."
            ),
            "per_split": all_stats,
        }, f, indent=2)
    print(f"\n[INFO] Conversion log saved to {LOG_PATH}")
    return results


if __name__ == "__main__":
    from src.data.masad_loader import load_all_splits
    splits = load_all_splits()
    run_pipeline(splits)
