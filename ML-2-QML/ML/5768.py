import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATExtended(nn.Module):
    """Enhanced classical model with depth‑wise separable convolutions and residual connections."""

    def __init__(self, num_features: int = 4, residual: bool = True, depth: int = 2) -> None:
        """
        Parameters
        ----------
        num_features : int
            Size of the final feature vector.
        residual : bool
            If True, add a residual connection after the convolutional backbone.
        depth : int
            Number of depth‑wise separable convolution blocks.
        """
        super().__init__()
        self.num_features = num_features
        self.residual = residual
        self.depth = depth

        # Build depth‑wise separable convolution backbone
        layers = []
        in_channels = 1
        for _ in range(depth):
            # Depth‑wise convolution
            layers.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=in_channels,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            # Point‑wise convolution
            out_channels = in_channels * 2
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_features),
        )
        self.norm = nn.BatchNorm1d(num_features)

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature map produced by the convolutional backbone."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        if self.residual and features.shape == x.shape:
            features = features + x
        flattened = features.view(features.shape[0], -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["QuantumNATExtended"]
