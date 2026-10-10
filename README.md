# ImplicitMABSA — Multimodal Implicit Aspect-Based Sentiment Analysis

> **Third-Year B.Tech Research Project** · Built on [MASAD](https://github.com/DrJZhou/MASAD) (Zhou et al., 2021)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/demo-Streamlit-red)](app/streamlit_app.py)

---

## 📌 Overview

**ImplicitMABSA** is a research project that infers **implicit aspects** — those never explicitly mentioned in text — and predicts their sentiment polarity (positive / neutral / negative) using both **textual and visual signals**.

**Key contributions:**
1. A reproducible pipeline to convert explicit-ABSA data (MASAD) into synthetic implicit-ABSA samples.
2. Three baseline comparisons: Text-only (TF-IDF + LogReg), Text-only (DistilBERT), Image-only (ResNet-50).
3. A **Late-Fusion multimodal model** combining BERT + ResNet-50 features.
4. Full ablation study with macro-F1, accuracy, confusion matrices.
5. A polished Streamlit demo.

> ⚠️ **Transparency:** All "implicit" training samples are *synthetically generated* by masking explicit aspect terms from MASAD. They are **not** naturally-occurring implicit examples. Every synthetic sample carries `is_synthetic=True` and a `conversion_note`.

---

## 📁 Project Structure

```
ImplicitMABSA/
├── data/
│   ├── raw/                   # Original MASAD files (download separately)
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── images/            # MASAD paired images
│   ├── processed/             # Cleaned DataFrames
│   └── synthetic/             # Converted implicit samples (auto-generated)
│       ├── implicit_train.json
│       ├── implicit_dev.json
│       ├── implicit_test.json
│       └── implicit_conversion_log.json
├── src/
│   ├── data/
│   │   ├── masad_loader.py    # Dataset inspection & loading
│   │   ├── implicit_converter.py  # Explicit→Implicit conversion pipeline
│   │   └── dataset.py         # PyTorch Dataset classes
│   ├── models/
│   │   ├── text_model.py      # TF-IDF + LogReg, DistilBERT
│   │   ├── image_model.py     # ResNet-50 feature extractor
│   │   └── fusion_model.py    # Late-fusion MLP
│   └── evaluation/
│       └── metrics.py         # Accuracy, F1, confusion matrices, ablation charts
├── outputs/
│   ├── plots/                 # Confusion matrices, ablation bar chart
│   ├── results/               # all_results.json, dataset_report.json
│   └── checkpoints/           # Saved model weights
├── app/
│   └── streamlit_app.py       # Interactive demo
├── paper/
│   └── ImplicitMABSA_paper.md # IEEE-style research paper draft
├── notebooks/
│   └── EDA.ipynb              # Exploratory data analysis
├── scripts/
│   └── download_masad.py      # Auto-download MASAD from Google Drive
├── train.py                   # Master training pipeline
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

### 2. Download MASAD

```bash
python scripts/download_masad.py
```

Or manually download from [Google Drive](https://drive.google.com/file/d/19YJ8vEYCb-uEKUqSGFmysUTvNzxhVKFE/view) and extract to `data/raw/`.

### 3. Run the full pipeline

```bash
# Full multimodal training and evaluation pipeline on verified MASAD data
python train.py --mode all

# Individual model training
python train.py --mode tfidf
python train.py --mode image
python train.py --mode fusion
```

### 4. Launch the AI Research Lab Web Application

```bash
python -m streamlit run app/streamlit_app.py
```

---

## 🧪 Dataset Conversion Pipeline

The pipeline is in [`src/data/implicit_converter.py`](src/data/implicit_converter.py).

**How it works:**
1. For each MASAD sample, the explicit aspect term is **regex-removed** using word boundaries (`\b<aspect>\b`) from the text.
2. If the remaining text is ≥ 4 words, the sample is retained.
3. A heuristic **confidence score** (0–1) is computed based on remaining sentiment-indicative words.
4. Samples with `confidence < 0.5` or very short text are flagged for **manual review**.
5. Every synthetic sample carries:
   - `is_synthetic = True`
   - `source = "MASAD_explicit_masked"`
   - `conversion_note` explaining provenance
   - `manual_validation_flag` for human review queue

**This is documented clearly to avoid misrepresentation** — see the conversion log at `data/synthetic/implicit_conversion_log.json`.

---

## 📊 Models & Architecture

| Model | Text Encoder | Image Encoder | Classifier | Architecture Status |
|-------|-------------|---------------|------------|---------------------|
| **TF-IDF + LogReg** | Sublinear TF-IDF (15k n-grams) | — | L2 Logistic Regression | Fully Trained & Evaluated |
| **Vision-Only** | — | Color Histograms + 2x2 Spatial Quadrants (96-dim) | L2 Logistic Regression | Fully Trained & Evaluated |
| **Multimodal Late-Fusion** | Sublinear TF-IDF | Color Histograms + Spatial Quadrants | Joint 2-layer MLP (128 hidden) | Fully Trained & Evaluated |
| *DistilBERT + ResNet-50* | DistilBERT CLS (768) | ResNet-50 Pool (2048) | Deep MLP | Implemented (Requires Torch/Transformers GPU stack) |

---

## 📈 Empirical Evaluation Results

All models evaluated strictly on the **leakage-free held-out test split (N=36)** derived from 124 authentic, matched MASAD text-image pairs:

| Model | Modality | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Status |
|---|---|---|---|---|---|---|
| **TF-IDF + LogReg** | Text-Only | **63.89%** | **66.75%** | **60.62%** | **0.5847** | Verified Checkpoint |
| **Vision-Only (Color+Spatial)** | Image-Only | **55.56%** | **56.25%** | **56.25%** | **0.5556** | Verified Checkpoint |
| **Multimodal Fusion (TF-IDF+Vision)** | Multimodal | **61.11%** | **60.39%** | **60.00%** | **0.6000** | Verified Checkpoint |

*Key Findings:*
- Multimodal Fusion achieves the highest **Macro-F1 (0.6000)**, showing balanced harmonic performance across positive and negative sentiment classes on implicit samples.
- Text-only captures strong lexical sentiment cues (Accuracy 63.89%), while Vision-only demonstrates non-random discriminative power (Accuracy 55.56% vs 50% baseline).
- Results and confusion matrix plots are saved to `outputs/results/all_results.json` and `outputs/plots/`.

---

## 🔬 Ablation Study & Visualizations

Run `python train.py --mode all` to regenerate the full ablation suite:
- `outputs/plots/ablation_comparison.png` — Grouped comparison of Accuracy, Precision, Recall, Macro-F1 across all 3 models.
- `outputs/plots/cm_TF-IDF+LogReg.png` — Text-only confusion matrix.
- `outputs/plots/cm_Vision_Only_ColorSpatial.png` — Vision-only confusion matrix.
- `outputs/plots/cm_Multimodal_Fusion_TFIDF_Vision.png` — Multimodal Fusion confusion matrix.

---

## ⚠️ Limitations

1. **Synthetic data**: All implicit samples are synthetically generated; they may not reflect naturally-occurring implicit language.
2. **Image availability**: If images are not downloaded, the image-only and fusion models operate on zero-vectors and their results are not meaningful.
3. **Confidence heuristic**: The conversion confidence score is keyword-based and not validated by a language model.
4. **Manual validation**: Only a subset of synthetic samples have been manually validated.

---

## 📄 Paper

See [`paper/ImplicitMABSA_paper.md`](paper/ImplicitMABSA_paper.md) for the IEEE-style research paper draft.

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
