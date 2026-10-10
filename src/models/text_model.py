"""
Text-Only Baseline Models
--------------------------
Two variants:
  1. TF-IDF + Logistic Regression (lightweight, interpretable)
  2. DistilBERT fine-tuned classifier (transformer-based)

Both expose a unified SKLearn-style interface:
    model.fit(X, y)
    model.predict(X)
    model.predict_proba(X)
"""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── TF-IDF Model ───────────────────────────────────────────────────────────────

class TFIDFTextModel:
    """
    Lightweight text-only baseline.
    Pipeline: TF-IDF vectorizer → Logistic Regression with L2 reg.
    """

    def __init__(
        self,
        max_features: int = 15_000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
            )),
        ])
        self.name = "TF-IDF + LogReg"
        self.classes_ = None

    def fit(self, texts: List[str], labels: List[int]):
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def get_token_attributions(self, text: str) -> List[tuple]:
        """
        Compute true token-level contributions based on TF-IDF weights and Logistic Regression coefficients.
        Returns list of (token, score) where positive score indicates positive sentiment contribution.
        """
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]

        # Transform single text
        X_vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        nonzero_indices = X_vec.nonzero()[1]

        # In binary classification, clf.coef_[0] corresponds to positive class (+1)
        # In multi-class, take difference between positive (+1) and negative (-1)
        if clf.coef_.shape[0] == 1:
            coefs = clf.coef_[0]
        else:
            # Match index of class 1 vs -1
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
            neg_idx = list(clf.classes_).index(-1) if -1 in clf.classes_ else (1 if clf.coef_.shape[0] > 1 else 0)
            coefs = clf.coef_[pos_idx] - clf.coef_[neg_idx]

        attributions = []
        for idx in nonzero_indices:
            term = feature_names[idx]
            weight = X_vec[0, idx]
            contribution = float(weight * coefs[idx])
            attributions.append((term, round(contribution, 4)))

        # Sort by absolute impact
        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return attributions

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[TF-IDF] Saved to {path}")

    @staticmethod
    def load(path: str) -> "TFIDFTextModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── DistilBERT Model ───────────────────────────────────────────────────────────

class DistilBERTTextModel:
    """
    DistilBERT fine-tuned for 3-class sentiment classification.
    Uses HuggingFace Transformers + PyTorch.
    Falls back gracefully if CUDA is not available.
    """

    LABEL2ID = {-1: 0, 0: 1, 1: 2}   # map sentiment ints → class indices
    ID2LABEL = {0: -1, 1: 0, 2: 1}
    NUM_LABELS = 3

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        epochs: int = 3,
        lr: float = 2e-5,
        random_state: int = 42,
    ):
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_available = True
        except ImportError:
            self.device = "cpu"
            self.torch_available = False

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.name = "DistilBERT"
        self._model = None
        self._tokenizer = None
        print(f"[DistilBERT] Torch available: {self.torch_available}, Using device: {self.device}")

    def _load_pretrained(self):
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self._model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.NUM_LABELS,
        ).to(self.device)

    def _tokenize(self, texts: List[str]):
        return self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def fit(self, texts: List[str], labels: List[int]):
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self._load_pretrained()

        # Map -1/0/1 labels → 0/1/2
        mapped = [self.LABEL2ID[int(l)] for l in labels]
        enc = self._tokenize(texts)
        dataset = TensorDataset(
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(mapped, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = AdamW(self._model.parameters(), lr=self.lr)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids, attn_mask, y = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                out = self._model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=y,
                )
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
            avg = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{self.epochs} — loss: {avg:.4f}")
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        import torch
        self._model.eval()
        all_preds = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenize(batch)
            with torch.no_grad():
                logits = self._model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend([self.ID2LABEL[p] for p in preds])
        return np.array(all_preds)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self._model.eval()
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenize(batch)
            with torch.no_grad():
                logits = self._model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        print(f"[DistilBERT] Saved to {path}")

    def load(self, path: str):
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        self._model = DistilBertForSequenceClassification.from_pretrained(path).to(
            self.device
        )
        return self
