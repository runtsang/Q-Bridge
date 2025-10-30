from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler that first encodes 2‑D image data with a CNN,
    projects to 4 features, and then maps to a 2‑class probability distribution
    via a small MLP.  This structure inherits the efficient feature extraction
    of Quantum‑NAT and the softmax sampling of SamplerQNN.
    """
    def __init__(self) -> None:
        super().__init__()
        # CNN encoder (identical to QFCModel.features)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected projection to 4 features
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        # Final classifier to 2 logits
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Probability distribution over 2 classes, shape (batch, 2).
        """
        bsz = x.shape[0]
        features = self.encoder(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)
        logits = self.classifier(out)
        probs = F.softmax(logits, dim=-1)
        return probs


def SamplerQNN() -> HybridSamplerQNN:
    """
    Compatibility wrapper that mirrors the original SamplerQNN function.
    Returns an instance of the new HybridSamplerQNN class.
    """
    return HybridSamplerQNN()


__all__ = ["HybridSamplerQNN", "SamplerQNN"]
