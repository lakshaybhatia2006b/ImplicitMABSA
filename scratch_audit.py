"""Quick audit script for ImplicitMABSA project."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))

# --- Test data loading ---
print("=== DATA LOADING AUDIT ===")
from src.data.masad_loader import load_all_splits, inspect_dataset, _build_image_index
splits = load_all_splits()
print('Splits loaded:', list(splits.keys()))
for k, v in splits.items():
    labels = v["label"].unique().tolist() if "label" in v.columns else None
    nan_labels = int(v["label"].isna().sum()) if "label" in v.columns else None
    print(f'  {k}: {len(v)} rows, label unique={labels}, nan_labels={nan_labels}')
    if "text" in v.columns:
        print(f'     sample text: {repr(v["text"].iloc[0][:80])}')
    if "image_id" in v.columns:
        print(f'     sample image_id: {repr(v["image_id"].iloc[0])}')

img_idx = _build_image_index()
print(f'\nImage index size: {len(img_idx)}')
if img_idx:
    sample_keys = list(img_idx.keys())[:5]
    for k in sample_keys:
        print(f'  {k} -> {img_idx[k]}')

# --- Test model checkpoints ---
print("\n=== MODEL CHECKPOINTS ===")
import pickle
ckpt_dir = Path("outputs/checkpoints")
for ckpt in ckpt_dir.glob("*.pkl"):
    try:
        with open(ckpt, "rb") as f:
            obj = pickle.load(f)
        print(f"  {ckpt.name}: Loaded OK, type={type(obj).__name__}")
        if hasattr(obj, 'classes_'):
            print(f"    classes_={obj.classes_}")
    except Exception as e:
        print(f"  {ckpt.name}: FAILED - {e}")

# --- Test inference ---
print("\n=== INFERENCE TEST (TF-IDF) ===")
try:
    from src.models.text_model import TFIDFTextModel
    model = TFIDFTextModel.load("outputs/checkpoints/tfidf_model.pkl")
    test_texts = ["This is amazing and beautiful!", "Terrible and awful experience."]
    preds = model.predict(test_texts)
    probs = model.predict_proba(test_texts)
    print(f"  Predictions: {preds}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Classes: {model.classes_}")
    attrs = model.get_token_attributions(test_texts[0])
    print(f"  Token attributions (first 3): {attrs[:3]}")
except Exception as e:
    print(f"  TF-IDF inference FAILED: {e}")

# --- Test vision inference ---
print("\n=== INFERENCE TEST (Vision) ===")
try:
    from src.models.lightweight_multimodal import LightweightVisionModel, find_image_file, get_image_cache
    cache = get_image_cache()
    print(f"  Vision image cache size: {len(cache)}")
    model = LightweightVisionModel.load("outputs/checkpoints/lightweight_vision_model.pkl")
    # Pick first real image ID from cache
    if cache:
        first_id = list(cache.keys())[0]
        preds = model.predict([first_id, "nonexistent_fake_id"])
        probs = model.predict_proba([first_id, "nonexistent_fake_id"])
        print(f"  Predictions for [{first_id}, fake]: {preds}")
        print(f"  Proba shape: {probs.shape}")
    else:
        print("  No images in cache - cannot test vision")
except Exception as e:
    import traceback
    print(f"  Vision inference FAILED: {e}")
    traceback.print_exc()

# --- Test fusion inference ---
print("\n=== INFERENCE TEST (Fusion) ===")
try:
    from src.models.lightweight_multimodal import LightweightMultimodalFusionModel
    model = LightweightMultimodalFusionModel.load("outputs/checkpoints/lightweight_fusion_model.pkl")
    if cache:
        first_id = list(cache.keys())[0]
        # Need text_pipeline to be set in the model
        preds = model.predict(["good experience"], [first_id])
        print(f"  Fusion prediction: {preds}")
    else:
        print("  No images - skipping fusion test")
except Exception as e:
    import traceback
    print(f"  Fusion inference FAILED: {e}")
    traceback.print_exc()

# --- Test evaluation results ---
print("\n=== EVALUATION RESULTS ===")
import json
results_file = Path("outputs/results/all_results.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    for model_name, metrics in results.items():
        print(f"  {model_name}: acc={metrics.get('accuracy')}, f1={metrics.get('macro_f1')}")

# --- Check implicit conversion ---
print("\n=== IMPLICIT CONVERSION CHECK ===")
syn_file = Path("data/synthetic/implicit_train.json")
if syn_file.exists():
    with open(syn_file) as f:
        syn_data = json.load(f)
    print(f"  Synthetic train samples: {len(syn_data)}")
    if syn_data:
        s = syn_data[0]
        print(f"  Sample keys: {list(s.keys())}")
        print(f"  is_synthetic: {s.get('is_synthetic')}")
        print(f"  source: {s.get('source')}")
        print(f"  masked_text: {repr(s.get('masked_text','')[:80])}")

print("\n=== AUDIT COMPLETE ===")
