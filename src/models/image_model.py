"""
Image-Only Baseline Model
--------------------------
Uses a pretrained ResNet-50 (or MobileNetV2) as a frozen feature extractor.
The final classification head is trained on top of pooled image features.

If images are unavailable, the model uses zero-vectors as placeholders and
prints a clear warning — it does NOT invent or hallucinate image data.
"""

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "images"
SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".gif"]
IMG_SIZE = (224, 224)   # ResNet / MobileNet expected input
FEATURE_DIM = 2048      # ResNet-50 pool layer output


def _find_image(image_id) -> Optional[Path]:
    """Locate the image file for a given image_id."""
    if image_id is None or (isinstance(image_id, float) and np.isnan(image_id)):
        return None
    image_dirs = [IMAGE_DIR]
    if RAW_DIR.exists():
        for p in RAW_DIR.rglob("image*"):
            if p.is_dir() and p not in image_dirs:
                image_dirs.append(p)

    for idir in image_dirs:
        if not idir.exists():
            continue
        for ext in ["", *SUPPORTED_EXTS]:
            p = idir / f"{image_id}{ext}"
            if p.exists() and p.is_file():
                return p
    return None


def load_image_tensor(image_id, device=None):
    """
    Load and preprocess a single image into a normalised PyTorch tensor.
    Returns (tensor, found: bool).
    If the image file is not found, returns a zero tensor and found=False.
    """
    import torch
    from torchvision import transforms
    from PIL import Image as PILImage

    device = device or torch.device("cpu")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    path = _find_image(image_id)
    if path is None:
        # Return zero tensor — NOT a fabricated image
        return torch.zeros(3, 224, 224).to(device), False

    try:
        img = PILImage.open(path).convert("RGB")
        return preprocess(img).to(device), True
    except Exception as e:
        warnings.warn(f"Could not load image {path}: {e}")
        return torch.zeros(3, 224, 224).to(device), False


class ImageModel:
    """
    ResNet-50 feature extractor + MLP classifier.
    
    When images are missing, this model operates on zero-vectors.
    Results in this case are meaningless and reported as such.
    """

    def __init__(
        self,
        backbone: str = "resnet50",  # or "mobilenet_v2"
        hidden_dim: int = 256,
        num_classes: int = 3,
        batch_size: int = 16,
        epochs: int = 5,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        self.backbone_name = backbone
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
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
        self.name = f"Image ({backbone})"
        self._backbone = None
        self._head = None
        self.images_available = IMAGE_DIR.exists()

        if not self.images_available:
            print(
                "[WARNING] Image directory not found. Image model will use "
                "zero-vectors as placeholders. Results will be at chance level "
                "and should NOT be used for any real comparison."
            )

    def _build_backbone(self):
        import torch.nn as nn
        from torchvision import models

        if self.backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove final FC
        elif self.backbone_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        # Freeze backbone weights
        for param in backbone.parameters():
            param.requires_grad = False

        self._backbone = backbone.to(self.device)
        self._feature_dim = feature_dim

        # Classification head
        self._head = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.num_classes),
        ).to(self.device)

    def _extract_features_batch(self, image_ids: List) -> "torch.Tensor":
        import torch
        tensors = []
        for img_id in image_ids:
            t, _ = load_image_tensor(img_id, self.device)
            tensors.append(t)
        batch = torch.stack(tensors)  # (B, 3, 224, 224)
        with torch.no_grad():
            feats = self._backbone(batch)  # (B, feature_dim)
        return feats

    # Label mapping: model outputs 0/1/2, maps to -1/0/1
    LABEL2ID = {-1: 0, 0: 1, 1: 2}
    ID2LABEL = {0: -1, 1: 0, 2: 1}

    def fit(self, image_ids: List, labels: List[int]):
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        torch.manual_seed(self.random_state)
        self._build_backbone()
        self._backbone.eval()  # Keep backbone frozen

        optimizer = Adam(self._head.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        mapped = torch.tensor(
            [self.LABEL2ID[int(l)] for l in labels], dtype=torch.long
        )

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(image_ids), self.batch_size):
                batch_ids = image_ids[i : i + self.batch_size]
                batch_labels = mapped[i : i + self.batch_size].to(self.device)

                feats = self._extract_features_batch(batch_ids)
                self._head.train()
                optimizer.zero_grad()
                logits = self._head(feats)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{self.epochs} — loss: {avg:.4f}")
        return self

    def predict(self, image_ids: List) -> np.ndarray:
        import torch
        preds = []
        self._head.eval()
        self._backbone.eval()
        with torch.no_grad():
            for i in range(0, len(image_ids), self.batch_size):
                batch_ids = image_ids[i : i + self.batch_size]
                feats = self._extract_features_batch(batch_ids)
                logits = self._head(feats)
                batch_preds = logits.argmax(dim=-1).cpu().numpy()
                preds.extend([self.ID2LABEL[p] for p in batch_preds])
        return np.array(preds)

    def predict_proba(self, image_ids: List) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        all_probs = []
        self._head.eval()
        self._backbone.eval()
        with torch.no_grad():
            for i in range(0, len(image_ids), self.batch_size):
                batch_ids = image_ids[i : i + self.batch_size]
                feats = self._extract_features_batch(batch_ids)
                probs = F.softmax(self._head(feats), dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def save(self, path: str):
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone_state": self._backbone.state_dict(),
            "head_state": self._head.state_dict(),
            "config": {
                "backbone": self.backbone_name,
                "hidden_dim": self.hidden_dim,
                "num_classes": self.num_classes,
            },
        }, path)
        print(f"[ImageModel] Saved to {path}")
