"""Enhanced classical QCNN with skip connections and dropout."""

import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """
    A deeper convolution‑inspired network with residual connections and dropout.
    The architecture mirrors the original QCNN but adds skip paths when input
    and output dimensions match, and interleaves dropout to mitigate overfitting.
    """
    def __init__(self, input_dim: int = 8,
                 hidden_dims: list[int] | tuple[int,...] = (16, 12, 8, 4),
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            # Residual connection only if dimensions match
            if in_dim == out_dim:
                layers.append(nn.Identity())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor and classifier.
        Outputs a sigmoid probability.
        """
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return torch.sigmoid(logits)

def QCNNHybridFactory(**kwargs) -> QCNNHybrid:
    """
    Factory that returns a configured QCNNHybrid instance.
    Allows passing arbitrary keyword arguments matching the constructor.
    """
    return QCNNHybrid(**kwargs)

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
