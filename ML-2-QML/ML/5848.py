from __future__ import annotations

import torch
import torch.nn as nn

class HybridEstimatorQNN(nn.Module):
    """Hybrid classical estimator with optional CNN encoder and configurable output dimensionality.

    The network consists of:
    - An optional convolutional encoder that extracts spatial features.
    - A linear block that maps the encoded features to the target space.
    - Batch‑norm and dropout for regularisation.
    The forward method accepts either raw 2‑D inputs (shape [N, 2]) or image‑like tensors
    (shape [N, C, H, W]).  When ``use_cnn=True`` the input is passed through the CNN
    encoder before the linear head.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        use_cnn: bool = False,
        cnn_channels: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_cnn = use_cnn
        self.output_dim = output_dim

        if use_cnn:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(cnn_channels, 2 * cnn_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Assuming 28x28 input (MNIST‑like) → 7x7 feature map
            conv_out_features = 2 * cnn_channels * 7 * 7
            self.head = nn.Sequential(
                nn.Linear(conv_out_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, output_dim),
            )

        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            # Expect shape [N, 1, H, W]
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
        x = self.head(x)
        return self.norm(x)

__all__ = ["HybridEstimatorQNN"]
