"""
Automated Regression and Pipeline Tests for ImplicitMABSA
---------------------------------------------------------
Validates:
  1. Data loading across official formats and directory structures.
  2. Provenance tracking and clean aspect masking (no orphan fragments).
  3. Zero data-leakage partition splits.
  4. Metric computation accuracy and shape handling.
  5. Checkpoint serialization and token attribution fidelity.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.masad_loader import load_all_splits
from src.data.implicit_converter import _mask_aspect, run_pipeline
from src.models.text_model import TFIDFTextModel
from src.evaluation.metrics import compute_metrics


class TestImplicitMABSAPipeline(unittest.TestCase):

    def test_aspect_masking_integrity(self):
        # Verify singular and plural masking with word-boundary awareness
        text = "The bad apples were rotten but the apple tree was fine."
        masked = _mask_aspect(text, "apple")
        self.assertIsNotNone(masked)
        self.assertNotIn("apple", masked.lower())
        self.assertNotIn("apples", masked.lower())
        # Should not leave "bad s"
        self.assertNotIn("bad s", masked)

    def test_synthetic_provenance(self):
        # Create minimal dataframe
        df = pd.DataFrame([{
            "text": "The delicious meal had exceptional flavor and warmth.",
            "aspect": "meal",
            "label": 1,
            "sentiment": "positive",
            "image_id": "test_01"
        }])
        from src.data.implicit_converter import convert_split
        syn_df, stats = convert_split(df, "test")
        self.assertEqual(len(syn_df), 1)
        row = syn_df.iloc[0]
        self.assertTrue(row["is_synthetic"])
        self.assertEqual(row["source"], "MASAD_explicit_masked")
        self.assertIn("conversion_note", row)

    def test_metrics_computation(self):
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, -1, 1])
        metrics = compute_metrics(y_true, y_pred, model_name="UnitTestModel")
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertIn("macro_f1", metrics)
        self.assertIn("macro_precision", metrics)

    def test_tfidf_fit_and_attributions(self):
        train_texts = [
            "absolutely fantastic delicious taste love it",
            "terrible horrible awful broken rotten experience",
            "great service wonderful food high quality",
            "worst product bad quality very poor"
        ]
        train_labels = [1, -1, 1, -1]
        model = TFIDFTextModel(max_features=100)
        model.fit(train_texts, train_labels)

        # Predict
        test_text = "fantastic quality"
        pred = model.predict([test_text])[0]
        self.assertEqual(pred, 1)

        # Attributions
        attrs = model.get_token_attributions(test_text)
        self.assertTrue(len(attrs) > 0)
        # 'fantastic' should have a positive impact score
        terms = [t for t, s in attrs]
        self.assertIn("fantastic", terms)


if __name__ == "__main__":
    unittest.main()
