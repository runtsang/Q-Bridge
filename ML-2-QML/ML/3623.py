"""Classical hybrid model combining CNN encoder with dual heads for classification and regression."""
import torch
import torch.nn as nn

class QFCModel(nn.Module):
    """
    Unified classical model with a CNN encoder followed by two parallel fully‑connected heads.
    The classification head outputs 4 logits; the regression head outputs a single scalar.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Feature map size after pooling: 7×7 → 16 channels
        self.flatten_dim = 16 * 7 * 7
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.flatten_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.norm_cls = nn.BatchNorm1d(num_classes)
        self.norm_reg = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass returning a dict with 'class_logits' and'regression'.
        """
        bsz = x.shape[0]
        features = self.encoder(x)
        flat = features.view(bsz, -1)
        cls_out = self.classifier(flat)
        reg_out = self.regressor(flat)
        return {
            "class_logits": self.norm_cls(cls_out),
            "regression": self.norm_reg(reg_out).squeeze(-1),
        }

__all__ = ["QFCModel"]
