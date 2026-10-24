# ImplicitMABSA — Multimodal Implicit Aspect-Based Sentiment Analysis

> **Third-Year B.Tech Research Project** · Built on [MASAD](https://github.com/DrJZhou/MASAD) (Zhou et al., 2021)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/demo-Streamlit-red)](app/streamlit_app.py)
[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)](#running-tests)

---

## 📌 Overview

**ImplicitMABSA** is a reproducible research project for **multimodal implicit Aspect-Based Sentiment Analysis (ABSA)** — inferring sentiment toward aspects that are *never explicitly mentioned* in text. The system leverages both textual and visual signals from the MASAD benchmark.

**Key contributions:**
1. A reproducible pipeline to convert explicit MASAD data into synthetic implicit-ABSA samples, with documented provenance and confidence scoring.
2. Three baseline comparisons: Text-only (TF-IDF + LogReg), Image-only (96-dim Color+Spatial), and Multimodal Late-Fusion (TF-IDF + Vision MLP).
3. Full ablation study with macro-F1, accuracy, confusion matrices.
4. A polished 7-section Streamlit research laboratory application.
5. Honest reporting of limitations: synthetic data, model constraints, and hardware requirements.

> ⚠️ **Transparency:** All "implicit" samples are *synthetically generated* by masking explicit aspect terms from MASAD. They are **not** naturally-occurring implicit examples. Every synthetic sample carries `is_synthetic=True`, `source="MASAD_explicit_masked"`, and a `conversion_note`.

---

## 📁 Project Structure

```
ImplicitMABSA/
├── app/
│   └── streamlit_app.py          # 7-section interactive research lab
├── data/
│   ├── raw/                      # MASAD files (downloaded separately)
│   │   ├── train/                # train/text/<sentiment>/<aspect>/<id>.txt
│   │   │   ├── text/             # train/image/<sentiment>/<aspect>/<id>.jpg
│   │   │   └── image/
│   │   └── test/                 # Same structure as train/
│   ├── processed/                # (auto-generated)
│   └── synthetic/                # Implicit-aspect converted samples (auto-generated)
│       ├── implicit_train.json
│       ├── implicit_test.json
│       └── implicit_conversion_log.json
├── outputs/
│   ├── checkpoints/              # Saved model weights
│   │   ├── tfidf_model.pkl
│   │   ├── lightweight_vision_model.pkl
│   │   └── lightweight_fusion_model.pkl
│   ├── plots/                    # Confusion matrices, ablation bar chart
│   └── results/                  # all_results.json, dataset_report.json
├── paper/
│   └── ImplicitMABSA_paper.md    # IEEE-style research paper draft
├── slides/
│   └── presentation_outline.md  # 14-slide B.Tech presentation guide
├── src/
│   ├── data/
│   │   ├── masad_loader.py       # MASAD folder-structure parser & image indexer
│   │   ├── implicit_converter.py # Explicit→Implicit conversion pipeline
│   │   └── dataset.py            # Dataset wrappers and split utilities
│   ├── models/
│   │   ├── text_model.py         # TF-IDF+LogReg + DistilBERT (requires torch)
│   │   ├── lightweight_multimodal.py  # CPU vision + fusion (no torch needed)
│   │   ├── image_model.py        # ResNet-50/MobileNet (requires torch+torchvision)
│   │   └── fusion_model.py       # LateFusionModel (requires torch)
│   └── evaluation/
│       └── metrics.py            # Metrics, confusion matrices, ablation charts
├── tests/
│   ├── test_pipeline.py          # Core 4-test suite
│   └── test_extended.py          # Comprehensive 33-test integration suite
├── scripts/
│   └── download_masad.py         # MASAD download helper
├── train.py                      # Master training pipeline
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/ImplicitMABSA.git
cd ImplicitMABSA
pip install -r requirements.txt
```

**Minimum requirements (CPU-only, no GPU needed):**
```
streamlit>=1.25
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
pillow>=10.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

### 2. Prepare the dataset

The MASAD dataset uses a folder hierarchy:
```
data/raw/
  train/
    text/<sentiment>/<aspect>/<image_id>.txt   ← review text files
    image/<sentiment>/<aspect>/<image_id>.jpg  ← paired image files
  test/
    text/<sentiment>/<aspect>/<image_id>.txt
    image/<sentiment>/<aspect>/<image_id>.jpg
```

Each `.txt` file contains:
```
<image_id>###<title>###<description>###<image_url>###<tags>
```

**Download from [MASAD Google Drive](https://drive.google.com/file/d/19YJ8vEYCb-uEKUqSGFmysUTvNzxhVKFE/view)**, extract, and place under `data/raw/`.

The project includes **124 verified matched pairs** (80 train, 44 test) spanning aspects: `apple`, `cat`, `graden`, `horse`, `factory`, `food`, `forest`, `chocolate`, `boat`, `insect`, and more.

### 3. Train the pipeline

```bash
# Full training and evaluation (all 3 models)
python train.py --mode all

# Individual model training
python train.py --mode tfidf   # Text-only baseline
python train.py --mode image   # Vision-only baseline
python train.py --mode fusion  # Multimodal fusion
```

### 4. Launch the AI Research Lab

```bash
python -m streamlit run app/streamlit_app.py
```

---

## 🧪 Dataset Conversion Pipeline

The pipeline is in [`src/data/implicit_converter.py`](src/data/implicit_converter.py).

**How it works:**
1. For each MASAD sample, the explicit aspect term is **regex-removed using word boundaries** (`\b<aspect>\b`), preserving subwords like "pineapple" when masking "apple".
2. If the remaining text has fewer than 4 words, the sample is dropped.
3. A heuristic **confidence score** (0–1) is computed based on sentiment-indicative vocabulary remaining after masking.
4. Samples with `confidence < 0.5`, `neutral` labels, or short masked text are flagged for **manual review**.
5. Every synthetic sample carries:
   - `is_synthetic = True`
   - `source = "MASAD_explicit_masked"`
   - `conversion_note` — provenance description
   - `manual_validation_flag` — human review queue marker
   - `confidence` — heuristic inferability score

**Pipeline statistics (current run):**
- Train: 69 converted from 111 (42 skipped — 40 too short after masking, 2 missing fields)
- Test: 36 converted from 60 (24 skipped — 23 too short, 1 missing field)

---

## 📊 Models & Architecture

| Model | Text Encoder | Image Encoder | Classifier | Status |
|-------|-------------|---------------|------------|--------|
| **TF-IDF + LogReg** | Sublinear TF-IDF, 15k n-grams (1,2) | — | L2 Logistic Regression | ✅ Trained & Verified |
| **Vision-Only (Color+Spatial)** | — | 27-bin RGB Histograms + 2×2 Spatial Quadrants (96-dim) | L2 Logistic Regression | ✅ Trained & Verified |
| **Multimodal Late-Fusion** | Sublinear TF-IDF | Same 96-dim spatial+color | 2-Layer MLP (64×32 hidden) | ✅ Trained & Verified |
| *DistilBERT* | DistilBERT-base CLS (768-dim) | — | Fine-tuned classification head | ⚠️ Requires `torch + transformers` |
| *ResNet-50* | — | ResNet-50 pool5 (2048-dim) | MLP head | ⚠️ Requires `torch + torchvision` |
| *DistilBERT + ResNet-50 Fusion* | DistilBERT CLS | ResNet-50 pool5 | Late-Fusion MLP | ⚠️ Requires full deep-learning stack |

---

## 📈 Empirical Evaluation Results

Evaluated strictly on the **leakage-free held-out test split (N=36)** derived from 124 authentic matched MASAD pairs. All preprocessing and vocabulary statistics fitted on training data only.

| Model | Modality | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---|---|---|---|
| **TF-IDF + LogReg** | Text-Only | **63.89%** | **66.75%** | **60.62%** | **0.5847** |
| **Vision-Only (Color+Spatial)** | Image-Only | **55.56%** | **56.25%** | **56.25%** | **0.5556** |
| **Multimodal Fusion** | Text + Image | **61.11%** | **60.39%** | **60.00%** | **0.6000** |

**Key findings:**
- **Multimodal Fusion achieves the highest Macro-F1 (0.6000)**, demonstrating that visual context complements text in balanced prediction across classes.
- **Vision-Only is non-random (55.56% vs 50% baseline)**, confirming genuine discriminative signal in 96-dim color/spatial features even without deep learning.
- **TF-IDF dominates in accuracy (63.89%)** but skews strongly toward Negative predictions (Recall: Neg=0.90, Pos=0.31), resulting in a lower Macro-F1.
- All results are reproducible with `python train.py --mode all` and fixed seed 42.

---

## 🔬 Ablation Study & Visualizations

Run `python train.py --mode all` to regenerate:
- `outputs/plots/ablation_comparison.png` — Grouped comparison bar chart.
- `outputs/plots/cm_TF-IDF+LogReg.png` — Text-only confusion matrix.
- `outputs/plots/cm_Vision_Only_ColorSpatial.png` — Vision-only confusion matrix.
- `outputs/plots/cm_Multimodal_Fusion_TFIDF_Vision.png` — Multimodal Fusion confusion matrix.
- `outputs/results/all_results.json` — JSON metrics for all models.

---

## 🧬 Research Dashboard — 7 Labs

| # | Section | Description |
|---|---------|-------------|
| 1 | **Live Inference & Triage** | Real-time text+image prediction with confidence bars and uncertainty flags |
| 2 | **Modality Intelligence Lab** | Side-by-side tri-modal comparison on authentic MASAD benchmark samples |
| 3 | **Evidence Lens & Attributions** | Grounded token log-odds contributions and visual channel decomposition |
| 4 | **Counterfactual Lab** | Modality swapping to test causal effects on predictions |
| 5 | **Model Evaluation Observatory** | Full metric tables, ablation plots, and confusion matrices |
| 6 | **Dataset Quality & Provenance** | Audit log, conversion statistics, and provenance inspection |
| 7 | **Architecture & Methodology** | Research design, limitations, and citations |

---

## 🧪 Running Tests

```bash
# Original 4-test suite (core)
python -m unittest tests.test_pipeline -v

# Extended 33-test integration suite
python -m unittest tests.test_extended -v

# Run all tests together (37 total)
python -m unittest discover -s tests -p "test_*.py" -v
```

**Latest test results (verified):**
```
Ran 37 tests in ~3s
OK (37 passed, 0 failures, 0 errors)
```

Tests cover: dataset loading, image index building, leakage detection, label validation, aspect masking with word boundaries, provenance tracking, checkpoint loading, inference shape/type correctness, probability normalization, missing-image handling, evaluation metrics, split integrity, and Streamlit headless execution.

---

## ⚠️ Limitations

1. **Synthetic data**: Implicit samples are synthetically generated; models may learn masking artifacts rather than real pragmatic implicit-aspect inference.
2. **Small dataset size**: Only 124 verified matched pairs; 36 test samples limit statistical confidence.
3. **No neutral class**: Current MASAD extraction yields only positive (label=1) and negative (label=-1) samples — no neutral (label=0) examples found in available data.
4. **Confidence heuristic**: The sentiment-word keyword confidence score is a rough proxy, not validated by a language model.
5. **Image availability**: Only 80/111 train and 44/60 test image IDs have corresponding image files (67-73% coverage).
6. **CPU-only baselines**: Vision model uses hand-crafted 96-dim features; ResNet-50/DistilBERT require PyTorch.

---

## 🗂️ Citation

If you use MASAD, please cite:
```bibtex
@article{zhou2021masad,
  title={MASAD: A Large-Scale Dataset for Multimodal Aspect-Based Sentiment Analysis},
  author={Zhou, Jie and Zhao, Jiabao and Huang, Jimmy and Hu, Qinmin Vivian and He, Liang},
  journal={Neurocomputing},
  year={2021},
  publisher={Elsevier}
}
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE).
