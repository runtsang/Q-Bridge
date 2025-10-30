import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNNConv(nn.Module):
    """
    Classical hybrid estimator that emulates the quantum EstimatorQNN
    with a convolutional feature extractor followed by a deep
    fully‑connected regression head.

    The model is intentionally deeper than the seed to expose
    richer feature hierarchies and to demonstrate how classical
    scaling can complement quantum circuits.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        conv_out_channels: int = 4,
        hidden_dims: tuple[int,...] = (32, 16),
        dropout: float = 0.2,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.threshold = threshold
        self.bn = nn.BatchNorm2d(conv_out_channels)

        # Fully‑connected regression head
        layers = []
        in_features = conv_out_channels  # because kernel_size==2 gives 1x1 output
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features, dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(dropout))
            in_features = dim
        layers.append(nn.Linear(in_features, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 1, kernel_size, kernel_size)

        Returns:
            Tensor of shape (batch, 1) containing regression predictions.
        """
        # Convolution + threshold gating
        y = self.conv(x)
        y = torch.sigmoid(y - self.threshold)
        y = self.bn(y)
        y = y.view(y.size(0), -1)
        return self.head(y)

__all__ = ["EstimatorQNNConv"]
