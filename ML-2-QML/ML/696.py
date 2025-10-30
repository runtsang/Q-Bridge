"""Enhanced classical quanvolution architecture with configurable backbone.

The module defines two classes:
- QuanvolutionFilter: a lightweight conv‑based feature extractor that
  can be extended with arbitrary extra layers, dropout and batch‑norm.
- QuanvolutionClassifier: a classifier that uses the filter followed by
  a small MLP with optional hidden layer, batch‑norm, dropout and ReLU.
Both classes preserve the original API (input shape [N,1,28,28]) and
return log‑softmax logits over 10 classes.

This design allows easy experimentation with different
backbones while keeping training pipelines unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Convolutional filter with optional extra layers.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 4
        Number of output channels produced by the first conv.
    kernel_size : int, default 2
        Size of the convolution kernel.
    stride : int, default 2
        Stride of the convolution.
    extra_layers : Sequence[nn.Module], optional
        Additional modules applied after the conv (e.g., activation).
    dropout_prob : float, default 0.0
        Dropout probability applied after the conv and extra layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        extra_layers=None,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.extra = nn.Sequential(*extra_layers) if extra_layers else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass."""
        x = self.conv(x)
        x = self.extra(x)
        x = self.dropout(x)
        return x.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that stacks a quanvolution filter with a small MLP.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    num_classes : int, default 10
        Number of target classes.
    hidden_dim : int, default 128
        Size of the hidden linear layer.
    dropout_prob : float, default 0.5
        Dropout probability after the hidden layer.
    use_batchnorm : bool, default True
        Whether to insert a BatchNorm1d after the hidden linear layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout_prob: float = 0.5,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(
            in_channels=in_channels, out_channels=4, kernel_size=2, stride=2
        )
        layers = [nn.Linear(4 * 14 * 14, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass."""
        features = self.filter(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
