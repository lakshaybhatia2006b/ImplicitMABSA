"""
Extended Integration Tests for ImplicitMABSA
---------------------------------------------
Covers:
  1. Dataset loading and image matching (with real data)
  2. Label mapping and implicit conversion integrity
  3. Model checkpoint loading (all three saved checkpoints)
  4. Prediction output types, shape, and probability consistency
  5. Evaluation metrics correctness
  6. Missing/nonexistent image handling
  7. Streamlit AST validity
  8. End-to-end inference (text, vision, fusion)
"""

import sys
import unittest
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.masad_loader import (
    load_all_splits, _build_image_index, _encode_sentiment
)
from src.data.implicit_converter import _mask_aspect, convert_split
from src.models.text_model import TFIDFTextModel
from src.models.lightweight_multimodal import (
    LightweightVisionModel,
    LightweightMultimodalFusionModel,
    find_image_file,
    get_image_cache,
    extract_visual_features_from_image,
)
from src.evaluation.metrics import compute_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

TFIDF_CKPT = ROOT / "outputs/checkpoints/tfidf_model.pkl"
VISION_CKPT = ROOT / "outputs/checkpoints/lightweight_vision_model.pkl"
FUSION_CKPT = ROOT / "outputs/checkpoints/lightweight_fusion_model.pkl"


def _small_tfidf():
    """Return a tiny trained TF-IDF model for fast tests."""
    train_texts = [
        "absolutely fantastic delicious taste love it",
        "terrible horrible awful broken rotten experience",
        "great service wonderful food high quality",
        "worst product bad quality very poor",
        "amazing beautiful superb excellent perfect",
        "ugly disgusting dirty foul smelly trash",
    ]
    train_labels = [1, -1, 1, -1, 1, -1]
    m = TFIDFTextModel(max_features=200)
    m.fit(train_texts, train_labels)
    return m


# ════════════════════════════════════════════════════════════════════════════

class TestDatasetLoading(unittest.TestCase):
    """Verify MASAD folder-structure loading and image matching."""

    def test_splits_load_successfully(self):
        splits = load_all_splits()
        self.assertTrue(len(splits) > 0, "No splits loaded from data/raw/")

    def test_train_split_has_required_columns(self):
        splits = load_all_splits()
        if "train" not in splits:
            self.skipTest("No train split available")
        df = splits["train"]
        for col in ("text", "aspect", "sentiment", "label", "image_id"):
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_labels_are_valid(self):
        splits = load_all_splits()
        for split_name, df in splits.items():
            self.assertIn("label", df.columns)
            valid = {-1, 0, 1}
            actual = set(df["label"].dropna().astype(int).unique())
            for v in actual:
                self.assertIn(v, valid, f"Invalid label {v} in {split_name}")

    def test_no_null_text(self):
        splits = load_all_splits()
        for name, df in splits.items():
            null_texts = df["text"].isna().sum()
            self.assertEqual(null_texts, 0, f"{name}: {null_texts} null text rows")

    def test_image_index_builds(self):
        idx = _build_image_index()
        # If we have any data in data/raw, the index must be non-empty
        raw_dir = ROOT / "data" / "raw"
        if any(raw_dir.rglob("*.jpg")):
            self.assertGreater(len(idx), 0, "Image index is empty but .jpg files exist")

    def test_image_ids_resolve_for_matched_samples(self):
        """Check that at least some image IDs resolve to real files."""
        splits = load_all_splits()
        if not splits:
            self.skipTest("No splits")
        cache = get_image_cache()
        if not cache:
            self.skipTest("No images in cache")
        all_ids = []
        for df in splits.values():
            if "image_id" in df.columns:
                all_ids.extend(df["image_id"].dropna().astype(str).tolist())
        resolved = sum(1 for iid in all_ids if iid in cache)
        self.assertGreater(resolved, 0, "No image IDs resolved to real files")

    def test_no_test_train_overlap(self):
        """Verify image IDs in test are not in train (leakage check)."""
        splits = load_all_splits()
        if "train" not in splits or "test" not in splits:
            self.skipTest("Need both train and test splits")
        train_ids = set(splits["train"]["image_id"].dropna().astype(str))
        test_ids = set(splits["test"]["image_id"].dropna().astype(str))
        overlap = train_ids & test_ids
        self.assertEqual(len(overlap), 0,
            f"Data leakage: {len(overlap)} image IDs appear in both train and test splits")


# ════════════════════════════════════════════════════════════════════════════

class TestImplicitConversion(unittest.TestCase):
    """Verify aspect masking correctness and provenance tracking."""

    def test_aspect_masking_basic(self):
        text = "The delicious food at this restaurant was extraordinary."
        masked = _mask_aspect(text, "food")
        self.assertIsNotNone(masked)
        self.assertNotIn("food", masked.lower())

    def test_aspect_masking_word_boundary(self):
        # 'apple' should not match inside 'pineapple'
        text = "I loved the pineapple smoothie but hate bad apple juice."
        masked = _mask_aspect(text, "apple")
        self.assertIsNotNone(masked)
        # 'pineapple' should survive; only standalone 'apple' is removed
        self.assertIn("pineapple", masked.lower())
        self.assertNotIn(" apple ", masked.lower())

    def test_aspect_masking_plural(self):
        text = "The bad apples were rotten but the apple tree was fine."
        masked = _mask_aspect(text, "apple")
        self.assertIsNotNone(masked)
        self.assertNotIn("apple", masked.lower())
        self.assertNotIn("apples", masked.lower())

    def test_aspect_masking_too_short(self):
        # After masking, text should be None when < MIN_WORDS
        text = "Bad food"
        masked = _mask_aspect(text, "food")
        # "Bad " is < 4 words -> should return None
        self.assertIsNone(masked)

    def test_conversion_provenance(self):
        df = pd.DataFrame([{
            "text": "The fresh garden salad was crisp and delightful.",
            "aspect": "salad",
            "label": 1,
            "sentiment": "positive",
            "image_id": "img_001"
        }])
        syn_df, stats = convert_split(df, "test_prov")
        self.assertEqual(len(syn_df), 1)
        row = syn_df.iloc[0]
        self.assertTrue(row["is_synthetic"])
        self.assertEqual(row["source"], "MASAD_explicit_masked")
        self.assertIn("conversion_note", row)
        self.assertNotIn("salad", row["masked_text"].lower())
        self.assertEqual(row["label"], 1)  # Original label preserved

    def test_conversion_skips_missing_fields(self):
        df = pd.DataFrame([
            {"text": "", "aspect": "food", "label": 1, "sentiment": "positive", "image_id": "x"},
            {"text": "Great tasting dish overall", "aspect": "", "label": 1, "sentiment": "positive", "image_id": "y"},
        ])
        syn_df, stats = convert_split(df, "test_skip")
        self.assertEqual(stats["skipped"], 2)
        self.assertEqual(len(syn_df), 0)


# ════════════════════════════════════════════════════════════════════════════

class TestModelCheckpoints(unittest.TestCase):
    """Verify all three checkpoints load and return correct types."""

    @unittest.skipUnless(TFIDF_CKPT.exists(), "TF-IDF checkpoint not found")
    def test_tfidf_checkpoint_loads(self):
        model = TFIDFTextModel.load(str(TFIDF_CKPT))
        self.assertIsInstance(model, TFIDFTextModel)
        self.assertIsNotNone(model.classes_)

    @unittest.skipUnless(VISION_CKPT.exists(), "Vision checkpoint not found")
    def test_vision_checkpoint_loads(self):
        model = LightweightVisionModel.load(str(VISION_CKPT))
        self.assertIsInstance(model, LightweightVisionModel)
        self.assertIsNotNone(model.classes_)

    @unittest.skipUnless(FUSION_CKPT.exists(), "Fusion checkpoint not found")
    def test_fusion_checkpoint_loads(self):
        model = LightweightMultimodalFusionModel.load(str(FUSION_CKPT))
        self.assertIsInstance(model, LightweightMultimodalFusionModel)
        self.assertIsNotNone(model.classes_)


# ════════════════════════════════════════════════════════════════════════════

class TestModelInference(unittest.TestCase):
    """Verify prediction shapes, types, and probability normalization."""

    def test_tfidf_inference_small(self):
        model = _small_tfidf()
        texts = ["fantastic quality", "terrible disaster"]
        preds = model.predict(texts)
        probs = model.predict_proba(texts)
        self.assertEqual(len(preds), 2)
        self.assertEqual(probs.shape, (2, len(model.classes_)))
        for row in probs:
            self.assertAlmostEqual(row.sum(), 1.0, places=5)
        for p in preds:
            self.assertIn(p, model.classes_)

    @unittest.skipUnless(TFIDF_CKPT.exists(), "TF-IDF checkpoint not found")
    def test_tfidf_checkpoint_inference(self):
        model = TFIDFTextModel.load(str(TFIDF_CKPT))
        preds = model.predict(["amazing and beautiful", "terrible and broken"])
        probs = model.predict_proba(["amazing and beautiful", "terrible and broken"])
        self.assertEqual(len(preds), 2)
        self.assertEqual(probs.shape[0], 2)
        for p in preds:
            self.assertIn(p, model.classes_)

    @unittest.skipUnless(TFIDF_CKPT.exists(), "TF-IDF checkpoint required")
    def test_tfidf_token_attributions(self):
        model = TFIDFTextModel.load(str(TFIDF_CKPT))
        attrs = model.get_token_attributions("This is absolutely fantastic quality")
        self.assertIsInstance(attrs, list)
        if attrs:
            term, score = attrs[0]
            self.assertIsInstance(term, str)
            self.assertIsInstance(score, float)

    @unittest.skipUnless(VISION_CKPT.exists(), "Vision checkpoint required")
    def test_vision_inference_with_real_image(self):
        model = LightweightVisionModel.load(str(VISION_CKPT))
        cache = get_image_cache()
        if not cache:
            self.skipTest("No images in cache")
        img_id = list(cache.keys())[0]
        preds = model.predict([img_id])
        probs = model.predict_proba([img_id])
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], model.classes_)
        self.assertAlmostEqual(probs[0].sum(), 1.0, places=5)

    @unittest.skipUnless(VISION_CKPT.exists(), "Vision checkpoint required")
    def test_vision_inference_missing_image(self):
        """Missing image should not crash; model should return a prediction using zero-vector."""
        model = LightweightVisionModel.load(str(VISION_CKPT))
        preds = model.predict(["totally_nonexistent_image_id_xyz"])
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], model.classes_)

    @unittest.skipUnless(FUSION_CKPT.exists(), "Fusion checkpoint required")
    def test_fusion_inference(self):
        model = LightweightMultimodalFusionModel.load(str(FUSION_CKPT))
        cache = get_image_cache()
        img_input = list(cache.keys())[0] if cache else "fake_id"
        preds = model.predict(["The view was absolutely stunning"], [img_input])
        probs = model.predict_proba(["The view was absolutely stunning"], [img_input])
        self.assertEqual(len(preds), 1)
        self.assertIn(preds[0], model.classes_)
        self.assertAlmostEqual(probs[0].sum(), 1.0, places=5)

    def test_vision_feature_extraction_real_image(self):
        """Feature vector should be non-zero and correct length for real image."""
        cache = get_image_cache()
        if not cache:
            self.skipTest("No images available")
        img_path = list(cache.values())[0]
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path).convert("RGB")
        feats, is_real = extract_visual_features_from_image(pil_img)
        self.assertTrue(is_real)
        self.assertEqual(len(feats), 96)
        self.assertGreater(feats.sum(), 0.0)

    def test_vision_feature_extraction_missing(self):
        """Zero-vector returned for nonexistent image; is_real=False."""
        feats, is_real = extract_visual_features_from_image("totally_fake_id_xyz_999")
        self.assertFalse(is_real)
        self.assertEqual(len(feats), 96)
        self.assertAlmostEqual(feats.sum(), 0.0, places=5)


# ════════════════════════════════════════════════════════════════════════════

class TestEvaluationMetrics(unittest.TestCase):
    """Verify metric calculations are correct."""

    def test_perfect_accuracy(self):
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, 1, -1, -1])
        m = compute_metrics(y_true, y_pred, model_name="PerfectTest")
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)

    def test_chance_accuracy(self):
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, -1, 1])
        m = compute_metrics(y_true, y_pred, model_name="ChanceTest")
        self.assertAlmostEqual(m["accuracy"], 0.5)

    def test_metric_keys_present(self):
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, -1, 1])
        m = compute_metrics(y_true, y_pred, model_name="KeyTest")
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            self.assertIn(key, m)

    def test_metrics_are_in_range(self):
        y_true = np.array([1, -1, 1, -1])
        y_pred = np.array([-1, 1, 1, -1])
        m = compute_metrics(y_true, y_pred, model_name="RangeTest")
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            self.assertGreaterEqual(m[key], 0.0)
            self.assertLessEqual(m[key], 1.0)

    def test_saved_results_file_integrity(self):
        """Verify the persisted results file is valid JSON with expected structure."""
        results_path = ROOT / "outputs/results/all_results.json"
        if not results_path.exists():
            self.skipTest("all_results.json not found — run train.py first")
        with open(results_path) as f:
            data = json.load(f)
        self.assertGreater(len(data), 0, "Results file is empty")
        for model_name, metrics in data.items():
            for key in ("accuracy", "macro_f1"):
                self.assertIn(key, metrics, f"Missing '{key}' for model '{model_name}'")
                val = metrics[key]
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)


# ════════════════════════════════════════════════════════════════════════════

class TestSplitIntegrity(unittest.TestCase):
    """Verify data splits do not leak."""

    def test_split_dataframe_no_overlap(self):
        from src.data.dataset import split_dataframe
        n = 20
        df = pd.DataFrame({
            "text": [f"sample {i}" for i in range(n)],
            "label": [1 if i % 2 == 0 else -1 for i in range(n)],
            "image_id": [str(i) for i in range(n)]
        })
        train, val, test = split_dataframe(df, seed=42)
        # Verify sizes add up and no rows are shared (compare by image_id, which is unique)
        self.assertEqual(len(train) + len(val) + len(test), n,
                         "Total rows don't sum to original dataset size")
        train_ids = set(train["image_id"].tolist())
        val_ids = set(val["image_id"].tolist())
        test_ids = set(test["image_id"].tolist())
        self.assertEqual(len(train_ids & val_ids), 0, "train/val overlap detected")
        self.assertEqual(len(train_ids & test_ids), 0, "train/test overlap detected")
        self.assertEqual(len(val_ids & test_ids), 0, "val/test overlap detected")

    def test_synthetic_implicit_train_all_marked(self):
        """Every row in the synthetic train file must have is_synthetic=True."""
        syn_path = ROOT / "data/synthetic/implicit_train.json"
        if not syn_path.exists():
            self.skipTest("implicit_train.json not found")
        with open(syn_path) as f:
            data = json.load(f)
        for i, row in enumerate(data):
            self.assertTrue(
                row.get("is_synthetic") is True,
                f"Row {i} in implicit_train.json has is_synthetic != True"
            )
            self.assertEqual(row.get("source"), "MASAD_explicit_masked",
                f"Row {i} has wrong source: {row.get('source')}")


# ════════════════════════════════════════════════════════════════════════════

class TestStreamlitApp(unittest.TestCase):
    """Verify the Streamlit app compiles and the AppTest reports no exceptions."""

    def test_app_syntax(self):
        import ast
        app_path = ROOT / "app/streamlit_app.py"
        self.assertTrue(app_path.exists(), "streamlit_app.py not found")
        with open(app_path, encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            self.fail(f"Streamlit app has syntax error: {e}")

    def test_app_headless_run(self):
        from streamlit.testing.v1 import AppTest
        app_path = ROOT / "app/streamlit_app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0,
            f"Streamlit app raised exceptions: {[str(e) for e in at.exception]}")


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
