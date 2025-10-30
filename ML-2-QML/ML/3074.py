import torch
import torch.nn as nn
from typing import List

class HybridClassifier(nn.Module):
    """
    Classical backbone for classification tasks.
    Supports three modes:
    - 'classic': pure feed‑forward network.
    - 'hybrid': outputs feature embeddings for a quantum head.
    - 'quantum': placeholder for quantum integration (no quantum ops here).
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 3,
                 mode: str = "classic",
                 num_classes: int = 2):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.mode = mode

        # Build classical backbone
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        self.backbone = nn.Sequential(*layers)

        # Classification head
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            If mode == 'classic' or 'quantum': logits of shape (batch, num_classes).
            If mode == 'hybrid': feature embeddings of shape (batch, num_features).
        """
        features = self.backbone(x)
        if self.mode in {"classic", "quantum"}:
            return self.head(features)
        # hybrid mode returns features for downstream quantum processing
        return features

__all__ = ["HybridClassifier"]
