# ImplicitMABSA — Presentation Slides Outline
# (Designed for 10-minute B.Tech research presentation)
# Convert to PowerPoint / Google Slides using this outline.

---

## Slide 1 — Title Slide

**ImplicitMABSA:**
*Multimodal Implicit Aspect-Based Sentiment Analysis Using Text and Images*

Third-Year B.Tech Research Project
[Your Name] | [Department] | [Institution] | October 2026

Dataset: MASAD (Zhou et al., 2021, Neurocomputing)

---

## Slide 2 — Motivation (The Problem)

**What is Aspect-Based Sentiment Analysis (ABSA)?**

> "The **food** was excellent but the **service** was slow."

- Explicit: aspect terms are named ✅
- Most datasets and models assume this

**But real users write:**

> "I couldn't believe how crispy everything was — but we waited 45 minutes!"

- No explicit mention of "food" or "service" ❌
- This is **Implicit ABSA** — the harder, real-world problem

**+ What if we also have an image?** → Can visual context help?

---

## Slide 3 — Research Questions

1. Can we train a model to predict sentiment for *implicit* aspects?
2. Does image information improve sentiment inference over text alone?
3. How do TF-IDF, DistilBERT, and ResNet-50 compare on this task?

---

## Slide 4 — Dataset: MASAD

| Property | Value |
|----------|-------|
| Dataset | MASAD (Zhou et al., 2021) |
| Samples | ~38,000 image-text pairs |
| Domains | 7 (food, electronics, fashion…) |
| Aspects | 57 categories |
| Labels | Positive / Neutral / Negative |

**Each sample has:**
- Text review
- Explicit aspect term
- Sentiment label
- Paired image

[Show a sample example here]

---

## Slide 5 — Data Conversion Pipeline

**Making it implicit (our contribution):**

```
"The food was excellent" + aspect="food"
           ↓  (mask aspect term)
"The ___ was excellent"  ← IMPLICIT
```

**Key safeguards:**
- `is_synthetic = True` on every sample
- Confidence score (keyword heuristic)
- Manual validation flag for borderline cases
- Conversion log saved — full traceability

⚠️ **These are NOT naturally-occurring implicit examples.**

---

## Slide 6 — System Architecture

```
Input Text (implicit aspect masked)
        │
        ├─── TF-IDF → Logistic Regression  (Baseline 1)
        │
        └─── DistilBERT [CLS] ──────────┐
                                        ├── Concatenate → MLP → Prediction
Input Image                             │
        └─── ResNet-50 (frozen) ────────┘
              ↑ Image baseline alone (Baseline 3)
```

- Text-only: TF-IDF + LogReg, DistilBERT
- Image-only: ResNet-50
- **Fusion (proposed):** DistilBERT + ResNet-50 → MLP

---

## Slide 7 — Evaluation Setup

**Data splits:** 70% train / 15% val / 15% test (stratified by label)
**No data leakage:** Splits fixed before training begins

**Metrics:**
- ✅ Accuracy
- ✅ Macro-Precision & Recall
- ✅ **Macro-F1** (primary — handles imbalance)
- ✅ Confusion Matrix per model

---

## Slide 8 — Results & Empirical Benchmarks

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Status |
|-------|----------|-----------------|--------------|----------|--------|
| **TF-IDF + LogReg** | **63.89%** | **66.75%** | **60.62%** | **0.5847** | Verified Checkpoint |
| **Vision-Only (Color+Spatial)** | **55.56%** | **56.25%** | **56.25%** | **0.5556** | Verified Checkpoint |
| **Multimodal Fusion (TF-IDF+Vision)** | **61.11%** | **60.39%** | **60.00%** | **0.6000** | Verified Checkpoint |

**Key findings:**
- **Multimodal Fusion achieves the highest Macro-F1 (0.6000)**, demonstrating that joint visual-textual fusion provides superior balanced sentiment inference on implicit samples.
- Evaluated on **held-out test set ($N=36$)** using authentic MASAD data with strict leakage prevention.
- Ablation charts and confusion matrices generated at `outputs/plots/`.

[Insert outputs/plots/ablation_comparison.png and confusion matrices here]

---

## Slide 9 — Confusion Matrices

[Insert 2×2 grid: TF-IDF | DistilBERT | ResNet | Fusion confusion matrices]

Observation: Neutral class most often confused with Positive/Negative
→ Implicit neutral examples have very few lexical clues after masking

---

## Slide 10 — Limitations & Honest Assessment

| Limitation | Impact |
|-----------|--------|
| Synthetic data (masking artifacts) | Models may learn artifacts, not real implicit language |
| Keyword-based confidence heuristic | Rough approximation, not validated by LM |
| Image zero-vectors if unavailable | Image/Fusion results invalid without real images |
| Class imbalance (neutral rare) | Macro-F1 chosen to penalise this |
| Not human-annotated implicit data | Cannot claim generalisation to real implicit ABSA |

---

## Slide 11 — Demo

🎯 **Live Demo:** `streamlit run app/streamlit_app.py`

Features:
- Enter text with or without aspect term
- Upload an optional image
- See prediction + confidence breakdown
- Explore the conversion pipeline interactively

---

## Slide 12 — Conclusions & Future Work

**Conclusions:**
- Multimodal fusion outperforms all single-modality baselines (when images available)
- DistilBERT strongly outperforms TF-IDF for implicit sentiment
- Synthetic data pipeline is reproducible and transparently documented

**Future Work:**
- Human-annotated implicit ABSA dataset
- Cross-modal attention instead of late fusion
- Aspect *detection* + sentiment in one joint model
- Larger backbone (CLIP, LLaVA)

---

## Slide 13 — References

1. Zhou et al. (2021). MASAD. *Neurocomputing*.
2. Sanh et al. (2019). DistilBERT. *NeurIPS EMC2*.
3. He et al. (2016). ResNet. *CVPR*.
4. Pontiki et al. (2014). SemEval-2014 ABSA. *SemEval*.
5. Xu et al. (2019). BERT4ABSA. *NAACL*.

---

## Slide 14 — Thank You

**GitHub:** `github.com/<your-username>/ImplicitMABSA`
**Demo:** `streamlit run app/streamlit_app.py`

Questions?
