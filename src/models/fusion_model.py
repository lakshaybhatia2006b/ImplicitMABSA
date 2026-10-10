"""
Multimodal Fusion Model
------------------------
Combines text features (DistilBERT or TF-IDF) with image features (ResNet-50)
using a late-fusion MLP classifier.

Fusion Strategy:
  1. Extract text feature vector (from DistilBERT CLS token or TF-IDF LSA)
  2. Extract image feature vector (from ResNet-50 pooled layer)
  3. Concatenate → MLP → softmax

This is the proposed multimodal model compared against text-only and image-only baselines.
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np


class LateFusionModel:
    """
    Late-fusion multimodal model:
      [text_features] + [image_features] → MLP → sentiment class

    Parameters
    ----------
    text_dim   : dimensionality of text feature vector
    image_dim  : dimensionality of image feature vector (2048 for ResNet-50)
    hidden_dim : MLP hidden layer size
    num_classes: 3 (positive / neutral / negative)
    dropout    : dropout rate on MLP
    """

    LABEL2ID = {-1: 0, 0: 1, 1: 2}
    ID2LABEL = {0: -1, 1: 0, 2: 1}

    def __init__(
        self,
        text_dim: int = 768,       # DistilBERT hidden size
        image_dim: int = 2048,     # ResNet-50 pool5 output
        hidden_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_available = True
        except ImportError:
            self.device = "cpu"
            self.torch_available = False
        self.name = "Multimodal Fusion"
        self._mlp = None

    def _build_mlp(self):
        import torch.nn as nn
        self._mlp = nn.Sequential(
            nn.Linear(self.text_dim + self.image_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes),
        ).to(self.device)

    def fit(
        self,
        text_features: np.ndarray,
        image_features: np.ndarray,
        labels: List[int],
    ):
        """
        Parameters
        ----------
        text_features  : shape (N, text_dim)
        image_features : shape (N, image_dim)
        labels         : list of int in {-1, 0, 1}
        """
        import torch
        import torch.nn as nn
        from torch.optim import Adam
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self._build_mlp()

        X = np.concatenate([text_features, image_features], axis=1)
        y = np.array([self.LABEL2ID[int(l)] for l in labels])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = Adam(self._mlp.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._mlp.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self._mlp(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{self.epochs} — loss: {avg:.4f}")
        return self

    def predict(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        import torch
        X = np.concatenate([text_features, image_features], axis=1)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self._mlp.eval()
        with torch.no_grad():
            logits = self._mlp(X_t)
            preds = logits.argmax(dim=-1).cpu().numpy()
        return np.array([self.ID2LABEL[p] for p in preds])

    def predict_proba(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        X = np.concatenate([text_features, image_features], axis=1)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self._mlp.eval()
        with torch.no_grad():
            probs = F.softmax(self._mlp(X_t), dim=-1).cpu().numpy()
        return probs

    def save(self, path: str):
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._mlp.state_dict(), path)
        print(f"[FusionModel] Saved to {path}")

    def load(self, path: str):
        import torch
        self._build_mlp()
        self._mlp.load_state_dict(torch.load(path, map_location=self.device))
        return self
