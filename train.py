"""
ImplicitMABSA — Master Training Pipeline
------------------------------------------
Runs the full experimental pipeline:
  0. Load & inspect dataset
  1. Convert to implicit samples
  2. Prepare stratified splits (no data leakage)
  3. Train and evaluate:
       a. TF-IDF + LogReg  (text-only baseline)
       b. DistilBERT        (text-only strong baseline)
       c. ResNet-50         (image-only baseline)
       d. Fusion (BERT+ResNet) MLP
  4. Ablation study
  5. Save all results, plots, and checkpoints

Usage:
    python train.py --mode tfidf           # TF-IDF baseline only
    python train.py --mode bert            # DistilBERT only
    python train.py --mode image           # Image only
    python train.py --mode fusion          # Fusion model
    python train.py --mode all             # Full experiment (default)
    python train.py --mode ablation        # Only ablation summary chart
    python train.py --use_explicit         # Use original explicit ABSA data
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
except ImportError:
    pass


# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.data.masad_loader import load_all_splits, inspect_dataset
from src.data.implicit_converter import run_pipeline
from src.data.dataset import split_dataframe
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_ablation,
    save_all_results,
)


# ── Feature Extraction Helpers ────────────────────────────────────────────────

def extract_bert_features(texts, model_name="distilbert-base-uncased",
                           batch_size=32, max_length=128):
    """Extract CLS-token embeddings from DistilBERT (no fine-tuning)."""
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name).to(device)
    model.eval()

    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_feats.append(cls)
    return np.vstack(all_feats)


def extract_image_features(image_ids, backbone="resnet50", batch_size=16):
    """Extract ResNet-50 pooled features. Zero-vector if image missing."""
    import torch
    from torchvision import models, transforms
    from PIL import Image as PILImage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_DIR = Path("data/raw/images")
    SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".gif"]

    # Build backbone
    if backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        feat_dim = 2048
    else:
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        net.classifier = torch.nn.Identity()
        feat_dim = 1280

    net = net.to(device)
    net.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from src.data.dataset import _find_image

    def load_tensor(img_id):
        if img_id is None:
            return torch.zeros(3, 224, 224)
        p = _find_image(img_id)
        if p and p.exists():
            try:
                return preprocess(PILImage.open(p).convert("RGB"))
            except Exception:
                pass
        return torch.zeros(3, 224, 224)

    all_feats = []
    for i in range(0, len(image_ids), batch_size):
        batch = [load_tensor(iid) for iid in image_ids[i : i + batch_size]]
        batch_t = torch.stack(batch).to(device)
        with torch.no_grad():
            feats = net(batch_t).cpu().numpy()
        all_feats.append(feats)
    return np.vstack(all_feats)


# ── Pipeline Steps ─────────────────────────────────────────────────────────────

def load_data(use_explicit: bool = False):
    """Load MASAD data and optionally convert to implicit."""
    print("\n" + "="*60)
    print(" STEP 0: Load Dataset")
    print("="*60)
    splits = load_all_splits()
    if not splits:
        print("[ERROR] No data found. Please download MASAD to data/raw/")
        sys.exit(1)
    inspect_dataset(splits)

    if use_explicit:
        print("\n[INFO] Using original explicit ABSA data.")
        return splits

    print("\n" + "="*60)
    print(" STEP 1: Convert to Implicit Samples")
    print("="*60)
    synthetic = run_pipeline(splits)
    return synthetic


def prepare_splits(data: dict):
    """
    Prepare leakage-free train, val, and test splits.
    If 'test' split is provided by the dataset, it is kept strictly intact for final testing.
    Train split is partitioned into train (80%) and validation (20%) sets.
    """
    from sklearn.model_selection import train_test_split

    if "train" in data and "test" in data:
        train_df = data["train"].dropna(subset=["label"]).copy()
        test_df = data["test"].dropna(subset=["label"]).copy()
        train_df["label"] = train_df["label"].astype(int)
        test_df["label"] = test_df["label"].astype(int)

        stratify_train = train_df["label"] if len(train_df["label"].unique()) > 1 else None
        # Split train into train and val (80/20)
        train_split, val_split = train_test_split(
            train_df,
            test_size=0.20,
            random_state=SEED,
            stratify=stratify_train,
        )
        print(f"\n[INFO] Leakage-free split:")
        print(f"  Train: {len(train_split)} samples, class dist: {train_split['label'].value_counts().to_dict()}")
        print(f"  Val  : {len(val_split)} samples, class dist: {val_split['label'].value_counts().to_dict()}")
        print(f"  Test : {len(test_df)} samples (held-out), class dist: {test_df['label'].value_counts().to_dict()}")
        return train_split.reset_index(drop=True), val_split.reset_index(drop=True), test_df.reset_index(drop=True)
    else:
        # Fallback if only single split provided
        all_dfs = [df for df in data.values()]
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.dropna(subset=["label"])
        combined["label"] = combined["label"].astype(int)
        print(f"\n[INFO] Combined dataset: {len(combined):,} samples")
        train, val, test = split_dataframe(combined, seed=SEED)
        return train, val, test


def run_tfidf(train, test, results):
    from src.models.text_model import TFIDFTextModel

    print("\n" + "="*60)
    print(" Baseline 1: TF-IDF + Logistic Regression")
    print("="*60)
    text_col = "masked_text" if "masked_text" in train.columns else "text"
    model = TFIDFTextModel()
    model.fit(train[text_col].tolist(), train["label"].tolist())
    preds = model.predict(test[text_col].tolist())
    metrics = compute_metrics(test["label"].values, preds, model_name="TF-IDF+LogReg")
    plot_confusion_matrix(test["label"].values, preds, model_name="TF-IDF+LogReg")
    results["TF-IDF+LogReg"] = metrics
    model.save("outputs/checkpoints/tfidf_model.pkl")
    return results


def run_bert(train, test, results):
    from src.models.text_model import DistilBERTTextModel

    print("\n" + "="*60)
    print(" Baseline 2: DistilBERT Fine-Tuned")
    print("="*60)
    text_col = "masked_text" if "masked_text" in train.columns else "text"
    model = DistilBERTTextModel(epochs=3, batch_size=32)
    model.fit(train[text_col].tolist(), train["label"].tolist())
    preds = model.predict(test[text_col].tolist())
    metrics = compute_metrics(test["label"].values, preds, model_name="DistilBERT")
    plot_confusion_matrix(test["label"].values, preds, model_name="DistilBERT")
    results["DistilBERT"] = metrics
    model.save("outputs/checkpoints/distilbert_model/")
    return results


def run_lightweight_vision(train, test, results):
    from src.models.lightweight_multimodal import LightweightVisionModel

    print("\n" + "="*60)
    print(" Baseline 3a: Image-Only (Color+Spatial Visual Features)")
    print("="*60)
    model = LightweightVisionModel(random_state=SEED)
    model.fit(train["image_id"].tolist(), train["label"].tolist())
    preds = model.predict(test["image_id"].tolist())
    metrics = compute_metrics(test["label"].values, preds, model_name="Vision-Only (Color+Spatial)")
    plot_confusion_matrix(test["label"].values, preds, model_name="Vision_Only_ColorSpatial")
    results["Vision-Only (Color+Spatial)"] = metrics
    model.save("outputs/checkpoints/lightweight_vision_model.pkl")
    return results


def run_lightweight_fusion(train, test, tfidf_model, results):
    from src.models.lightweight_multimodal import LightweightMultimodalFusionModel

    print("\n" + "="*60)
    print(" Model 4a: Multimodal Late Fusion (TF-IDF Text + Visual Features MLP)")
    print("="*60)
    text_col = "masked_text" if "masked_text" in train.columns else "text"
    text_vec = tfidf_model.pipeline.named_steps["tfidf"]

    model = LightweightMultimodalFusionModel(random_state=SEED)
    model.fit(
        train[text_col].tolist(),
        train["image_id"].tolist(),
        train["label"].tolist(),
        text_pipeline=text_vec
    )
    preds = model.predict(test[text_col].tolist(), test["image_id"].tolist())
    metrics = compute_metrics(test["label"].values, preds, model_name="Multimodal Fusion (TF-IDF+Vision)")
    plot_confusion_matrix(test["label"].values, preds, model_name="Multimodal_Fusion_TFIDF_Vision")
    results["Multimodal Fusion (TF-IDF+Vision)"] = metrics
    model.save("outputs/checkpoints/lightweight_fusion_model.pkl")
    return results


def run_fusion(train, test, results):
    from src.models.fusion_model import LateFusionModel

    print("\n" + "="*60)
    print(" Model 4: Multimodal Late Fusion (BERT + ResNet-50)")
    print("="*60)
    text_col = "masked_text" if "masked_text" in train.columns else "text"

    print("[INFO] Extracting BERT features for train set...")
    train_text_feats = extract_bert_features(train[text_col].tolist())
    print("[INFO] Extracting BERT features for test set...")
    test_text_feats = extract_bert_features(test[text_col].tolist())

    print("[INFO] Extracting image features for train set...")
    train_img_feats = extract_image_features(train["image_id"].tolist())
    print("[INFO] Extracting image features for test set...")
    test_img_feats = extract_image_features(test["image_id"].tolist())

    model = LateFusionModel(
        text_dim=train_text_feats.shape[1],
        image_dim=train_img_feats.shape[1],
        epochs=10,
    )
    model.fit(train_text_feats, train_img_feats, train["label"].tolist())
    preds = model.predict(test_text_feats, test_img_feats)
    metrics = compute_metrics(
        test["label"].values, preds, model_name="Multimodal Fusion"
    )
    plot_confusion_matrix(
        test["label"].values, preds, model_name="Multimodal_Fusion"
    )
    results["Multimodal Fusion"] = metrics
    model.save("outputs/checkpoints/fusion_model.pt")
    return results


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ImplicitMABSA Training Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["tfidf", "bert", "image", "fusion", "all", "ablation"],
        default="all",
        help="Which model(s) to train.",
    )
    parser.add_argument(
        "--use_explicit",
        action="store_true",
        help="Use original explicit ABSA data instead of synthetic implicit data.",
    )
    parser.add_argument(
        "--high_confidence_only",
        action="store_true",
        default=False,
        help="Filter synthetic samples to confidence >= 0.6 only.",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  ImplicitMABSA -- Multimodal Implicit ABSA Pipeline")
    print("="*60)
    print(f"  Mode         : {args.mode}")
    print(f"  Use explicit : {args.use_explicit}")
    print(f"  Seed         : {SEED}")

    # Load data
    data = load_data(use_explicit=args.use_explicit)

    # Prepare splits
    train, val, test = prepare_splits(data)

    # Optionally filter low-confidence synthetic samples
    if not args.use_explicit and args.high_confidence_only:
        before = len(train)
        train = train[train.get("confidence", 1.0) >= 0.6]
        print(f"[INFO] High-confidence filter: {before} -> {len(train)} training samples")

    # Run experiments
    results = {}

    tfidf_model = None
    if args.mode in ("tfidf", "all"):
        results = run_tfidf(train, test, results)
        from src.models.text_model import TFIDFTextModel
        tfidf_model = TFIDFTextModel.load("outputs/checkpoints/tfidf_model.pkl")

    if args.mode in ("image", "all"):
        # Run lightweight vision baseline (verified CPU)
        results = run_lightweight_vision(train, test, results)
        # Deep learning ResNet baseline if torch is present
        try:
            import torch, torchvision
            from src.models.image_model import ImageModel
            print("\n[INFO] PyTorch detected: Running ResNet-50 vision baseline...")
            res_model = ImageModel(backbone="resnet50", epochs=5)
            res_model.fit(train["image_id"].tolist(), train["label"].tolist())
            res_preds = res_model.predict(test["image_id"].tolist())
            results["Image (ResNet-50)"] = compute_metrics(test["label"].values, res_preds, model_name="Image (ResNet-50)")
        except ImportError:
            pass

    if args.mode in ("fusion", "all") and tfidf_model is not None:
        # Run lightweight multimodal fusion model
        results = run_lightweight_fusion(train, test, tfidf_model, results)

    if args.mode in ("bert", "all"):
        try:
            import torch, transformers
            results = run_bert(train, test, results)
        except ImportError as e:
            print(f"\n[SKIP] DistilBERT requires torch & transformers ({e}). Skipping...")

    # ── Ablation chart ──
    if results:
        print("\n" + "="*60)
        print(" ABLATION STUDY")
        print("="*60)
        plot_ablation(results)
        save_all_results(results)

    print("\n[DONE] Training pipeline complete.")
    print("  Results: outputs/results/")
    print("  Plots  : outputs/plots/")
    print("  Models : outputs/checkpoints/")


if __name__ == "__main__":
    main()
