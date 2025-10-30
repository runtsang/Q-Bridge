"""Hybrid classical model combining convolutional feature extraction,
randomised quantum‑inspired feature mapping, and a scalable feed‑forward
classifier.  The architecture mirrors the quantum version but is fully
implementable with PyTorch, enabling quick prototyping and baseline
evaluation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuanvolutionNet(nn.Module):
    """
    Classical hybrid model that emulates quantum feature extraction using a
    random linear projection.  It contains:
      * A convolutional feature extractor (modeled after QFCModel).
      * A per‑patch random projection that behaves like a quantum kernel.
      * A configurable feed‑forward classifier.
    """

    def __init__(
        self,
        depth: int = 2,
        num_classes: int = 10,
        patch_size: int = 2,
        feature_dim: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim

        # Convolutional feature extractor (same as QFCModel)
        self.conv_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Random projection per 2x2 patch → feature_dim dimensions
        self.register_buffer(
            "random_proj",
            torch.randn(patch_size * patch_size, feature_dim),
        )

        # Classifier
        in_features = feature_dim * 14 * 14  # 14x14 patches from 28x28 input
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_features, in_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Scaling metadata
        self.encoding = list(range(in_features))
        self.weight_sizes = [
            layer.weight.numel() + layer.bias.numel()
            for layer in self.classifier
            if isinstance(layer, nn.Linear)
        ]
        self.observables = list(range(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Extract features with conv_extractor.
          2. Unfold into 2x2 patches, average across channels.
          3. Apply random projection per patch.
          4. Flatten and classify.
        """
        bsz = x.size(0)

        # Feature extraction
        feats = self.conv_extractor(x)  # (bsz, 16, 7, 7)

        # 2x2 patches: unfold then reshape
        patches = feats.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(bsz, 16, 14 * 14, self.patch_size * self.patch_size)

        # Average across the 16 feature maps (like avg_pool2d in QFCModel)
        patches = patches.mean(dim=1)  # (bsz, 14*14, 4)

        # Random projection per patch
        projected = torch.matmul(patches, self.random_proj)  # (bsz, 14*14, feature_dim)
        projected = projected.view(bsz, -1)  # flatten

        logits = self.classifier(projected)
        return F.log_softmax(logits, dim=-1)

    def get_classifier_meta(self):
        """
        Return the same metadata used by the quantum counterpart:
          - encoding indices
          - weight sizes per linear layer
          - observables (class indices)
        """
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["HybridQuanvolutionNet"]
