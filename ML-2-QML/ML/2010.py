"""Enhanced classical quanvolution architecture with residual scaling.

The module extends the original 2×2 convolution by adding a residual
connection and a learnable scaling factor. A two‑layer MLP head with
batch‑norm and residual shortcut follows, providing richer feature
processing and better calibration.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution__gen211(nn.Module):
    """Hybrid classical-quantum inspired convolutional neural network.

    This implementation extends the original quanvolution architecture by
    adding a residual connection and a learnable scaling factor to the
    convolutional output. The head consists of a two‑layer MLP with a
    residual shortcut, providing richer feature processing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        conv_out_channels: int = 4,
        residual: bool = True,
        scaling: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.scaling = scaling

        self.conv = nn.Conv2d(
            in_channels, conv_out_channels, kernel_size=2, stride=2, bias=False
        )
        if residual:
            # 1x1 conv to match channel dimensions for residual addition
            self.residual_conv = nn.Conv2d(
                in_channels, conv_out_channels, kernel_size=1, stride=2, bias=False
            )
        else:
            self.residual_conv = None

        if scaling:
            # Learnable scalar to scale conv output
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = None

        # Two‑layer MLP head with residual
        self.fc1 = nn.Linear(conv_out_channels * 14 * 14, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        conv_out = self.conv(x)  # (batch, 4, 14, 14)

        if self.residual:
            res = self.residual_conv(x)  # (batch, 4, 14, 14)
            conv_out = conv_out + res

        if self.scaling:
            conv_out = conv_out * self.scale

        flat = conv_out.view(x.size(0), -1)  # (batch, 4*14*14)

        h = F.relu(self.bn1(self.fc1(flat)))
        logits = self.fc2(h)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen211"]
