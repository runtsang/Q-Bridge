"""Combined classical QCNN and sampler network.

This module merges the convolution‑style feature extractor from
the original QCNNModel with a lightweight sampler head
inspired by SamplerQNN.  The network can be used as a drop‑in
replacement for either component while still providing a
single entry point for joint training or inference.
"""

import torch
from torch import nn
import torch.nn.functional as F

class QCNNGen142(nn.Module):
    """Classical hybrid network: QCNN feature extractor + sampler head."""
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – a deeper, ReLU‑activated version of QCNNModel
        self.feature_extractor = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 24), nn.ReLU(),
            nn.Linear(24, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        # Sampler head – similar to SamplerQNN but takes 8‑dim features
        self.sampler_head = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 8), nn.Tanh(),
            nn.Linear(8, 2),
        )
    def forward(self, x: torch.Tensor) -> dict:
        """Return both QCNN‑style logits and sampler probabilities."""
        features = self.feature_extractor(x)
        qcnn_logits = features[:, :1]          # first dimension as QCNN output
        qcnn_out = torch.sigmoid(qcnn_logits)
        sampler_logits = self.sampler_head(features)
        sampler_out = F.softmax(sampler_logits, dim=-1)
        return {"qcnn": qcnn_out, "sampler": sampler_out}

def QCNNGen142_factory() -> QCNNGen142:
    """Factory that returns a ready‑to‑use instance."""
    return QCNNGen142()

__all__ = ["QCNNGen142", "QCNNGen142_factory"]
