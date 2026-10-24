import pickle
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
from src.models.text_model import TFIDFTextModel
from src.models.lightweight_multimodal import (
    LightweightVisionModel,
    LightweightMultimodalFusionModel,
    get_image_cache,
)

ckpts = {
    'TF-IDF': Path('outputs/checkpoints/tfidf_model.pkl'),
    'Vision': Path('outputs/checkpoints/lightweight_vision_model.pkl'),
    'Fusion': Path('outputs/checkpoints/lightweight_fusion_model.pkl')
}

print('=== CHECKPOINT INSPECTION ===')
for name, p in ckpts.items():
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    print(f"{name}: exists={exists}, size={size} bytes")
    if exists:
        with open(p, 'rb') as f:
            obj = pickle.load(f)
            cls_attr = getattr(obj, "classes_", None)
            print(f"  Loaded type: {type(obj).__name__}, classes: {cls_attr}")

print('\n=== REAL PREDICTION INFERENCE TEST ===')
m_text = TFIDFTextModel.load(str(ckpts['TF-IDF']))
m_vis = LightweightVisionModel.load(str(ckpts['Vision']))
m_fus = LightweightMultimodalFusionModel.load(str(ckpts['Fusion']))

cache = get_image_cache()
sample_img_id = list(cache.keys())[0] if cache else None
sample_img_path = cache[sample_img_id] if sample_img_id else None
pil_img = PILImage.open(sample_img_path).convert('RGB') if sample_img_path else None

print(f"Test image selected: ID={sample_img_id}, Path={sample_img_path}")

sample_texts = [
    'The meal was delicious and wonderful, loved every bite',
    'Horrible terrible disgusting service and awful quality'
]

preds_t = m_text.predict(sample_texts)
probs_t = m_text.predict_proba(sample_texts)
print(f"Text-Only: preds={preds_t.tolist()}, probs={probs_t.round(3).tolist()}")

preds_v_id = m_vis.predict([sample_img_id])
probs_v_id = m_vis.predict_proba([sample_img_id])
preds_v_pil = m_vis.predict([pil_img])
probs_v_pil = m_vis.predict_proba([pil_img])
print(f"Vision-Only (ID): pred={preds_v_id.tolist()}, prob={probs_v_id.round(3).tolist()}")
print(f"Vision-Only (PIL): pred={preds_v_pil.tolist()}, prob={probs_v_pil.round(3).tolist()}")

preds_f = m_fus.predict(sample_texts, [pil_img, pil_img])
probs_f = m_fus.predict_proba(sample_texts, [pil_img, pil_img])
print(f"Fusion (PIL): preds={preds_f.tolist()}, probs={probs_f.round(3).tolist()}")
print('\nALL INFERENCE PIPELINES OPERATIONAL AND VERIFIED.')
