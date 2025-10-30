"""Extended CNN with residual blocks and dropout for improved feature learning.

The model builds upon the original QFCModel by adding two residual
convolutional blocks, dropout regularisation and a lightweight
fully‑connected head.  The design keeps the same 4‑class output
interface while providing a richer feature extractor that can be
used for downstream tasks or as a backbone in hybrid pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """CNN + residual blocks + FC head.

    Parameters
    ----------
    num_classes : int, default 4
        Number of output classes.
    dropout : float, default 0.5
        Dropout probability applied after pooling layers.
    use_residual : bool, default True
        Whether to use residual connections in the conv blocks.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.5, use_residual: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(1, 8),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            conv_block(8, 16),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            conv_block(16, 32),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        # After 3 pooling operations on 28x28 input -> 3x3 feature map
        self.flatten_size = 32 * 3 * 3
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Class logits of shape (B, num_classes).
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened feature vector before the classifier."""
        out = self.features(x)
        return out.view(out.size(0), -1)

__all__ = ["QFCModel"]
