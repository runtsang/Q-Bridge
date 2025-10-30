from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Iterable

class HybridLayer(nn.Module):
    """
    Classical hybrid layer combining convolution, fully‑connected,
    classifier, and sampler blocks.  Mirrors the functionality of the
    seed modules while allowing end‑to‑end PyTorch execution.
    """

    def __init__(self, conv_kernel: int = 2, n_features: int = 1, classifier_depth: int = 2):
        super().__init__()
        # Convolutional filter (drop‑in for quanvolution)
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel, bias=True)

        # Fully‑connected layer
        self.fc = nn.Linear(n_features, 1)

        # Classifier network
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(classifier_depth):
            layers.append(nn.Linear(in_dim, n_features))
            layers.append(nn.ReLU())
            in_dim = n_features
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

        # Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through conv → flatten → fc → classifier → sampler.
        Input shape: (batch, 1, H, W) for the convolutional stage.
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.classifier(x)
        probs = self.sampler(logits)
        return probs

    def run(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Convenience wrapper that accepts a dict with keys 'conv', 'fc',
        'classifier', and'sampler' and returns the corresponding outputs
        as PyTorch tensors.
        """
        outputs: Dict[str, Tensor] = {}
        # Convolution
        conv_out = self.conv(data["conv"])
        outputs["conv"] = conv_out.detach()
        # Fully‑connected
        flat = conv_out.view(conv_out.size(0), -1)
        fc_out = self.fc(flat)
        outputs["fc"] = fc_out.detach()
        # Classifier
        logits = self.classifier(fc_out)
        outputs["classifier"] = logits.detach()
        # Sampler
        probs = self.sampler(logits)
        outputs["sampler"] = probs.detach()
        return outputs

__all__ = ["HybridLayer"]
