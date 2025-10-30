"""QCNNPlus: classical autoencoder‑style QCNN with dropout.

This class implements an autoencoder that first compresses the 8‑dimensional
input into a 16‑dimensional latent space, then reconstructs the input
before a final classification head.  Dropout is applied after the encoder
to regularise the latent representation.  The architecture mirrors the
original QCNN but adds a learnable feature extractor and a dropout
regulariser, making it more expressive while still lightweight.

Typical usage::

    model = QCNNPlus()
    logits = model(x)  # x is a torch.Tensor of shape (batch, 8)

"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class QCNNPlus(nn.Module):
    """Autoencoder‑style QCNN with dropout regularisation."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        dropout: float = 0.3,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        # Encoder: linear layers with ReLU activations
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        # Dropout after encoding
        self.dropout = nn.Dropout(dropout)
        # Decoder: linear layers that reconstruct the original dimension
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )
        # Classification head
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through the autoencoder and classifier."""
        z = self.encoder(x)
        z = self.dropout(z)
        recon = self.decoder(z)
        logits = self.classifier(recon)
        return torch.sigmoid(logits)

def QCNNPlusFactory() -> QCNNPlus:
    """Return a freshly constructed :class:`QCNNPlus` instance."""
    return QCNNPlus()

__all__ = ["QCNNPlus", "QCNNPlusFactory"]
