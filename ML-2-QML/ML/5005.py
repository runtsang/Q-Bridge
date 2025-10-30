from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNATModel(nn.Module):
    """
    A hybrid classical neural network that extends the original Quantum‑NAT
    design with additional scaling, clipping and a choice between
    classification or regression heads.

    The network consists of:
      * a small convolutional feature extractor,
      * a linear projection with optional clipping of weights,
      * a Tanh activation followed by optional learnable scaling & shift
        inspired by the fraud‑detection photonic analogue,
      * a final task‑specific head (softmax for classification,
        linear for regression).
    """

    def __init__(
        self,
        num_channels: int = 1,
        conv_out_channels: int = 16,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        fc_hidden: int = 64,
        clip_weights: bool = False,
        clip_bound: float = 5.0,
        task: str = "classification",
        num_classes: int = 2,
        regression_dim: int = 1,
    ) -> None:
        super().__init__()

        self.clip_weights = clip_weights
        self.clip_bound = clip_bound

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, conv_out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Compute flattened size after two 2x2 poolings
        dummy_input = torch.zeros(1, num_channels, 28, 28)  # typical MNIST size
        with torch.no_grad():
            feat = self.features(dummy_input)
        flat_size = feat.numel()

        # Linear projection
        self.fc = nn.Sequential(
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        # Learnable scaling & shift (FraudDetection style)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("shift", torch.zeros(4))

        # Task head
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(4, num_classes),
                nn.LogSoftmax(dim=1),
            )
        elif task == "regression":
            self.head = nn.Linear(4, regression_dim)
        else:
            raise ValueError("task must be 'classification' or'regression'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Feature extraction
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)

        # Linear projection
        out = self.fc(flat)

        # Optional clipping of weights
        if self.clip_weights:
            with torch.no_grad():
                for param in self.fc.parameters():
                    param.clamp_(-self.clip_bound, self.clip_bound)

        # Normalization
        out = self.norm(out)

        # Scaling & shift
        out = out * self.scale + self.shift

        # Head
        return self.head(out)

__all__ = ["HybridNATModel"]
