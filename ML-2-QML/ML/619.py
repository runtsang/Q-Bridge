"""Enhanced classical QCNN with residual connections and regularisation.

The architecture mirrors the original QCNN but replaces the linear layers
with a shallow convolutional stack (1D Conv layers) and adds batch
normalisation, dropout and residual links.  This improves feature
propagation and reduces over‑fitting on small datasets.
"""

import torch
from torch import nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """Convolutional neural network inspired by the quantum circuit.

    The network processes 8‑dimensional input vectors through a
    sequence of Conv1d → BN → ReLU blocks, followed by pooling and a
    residual skip connection.  The final layer outputs a single
    probability with a sigmoid activation.
    """

    def __init__(self, in_channels: int = 1, seq_len: int = 8) -> None:
        super().__init__()
        # Feature extraction
        self.feature_map = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Residual projection
        self.res_proj = nn.Conv1d(16, 4, kernel_size=1)
        # Classification head
        self.head = nn.Linear(4 * seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Reshape to (batch, channels, seq_len)
        x = x.unsqueeze(1)  # (B, 1, 8)
        feat = self.feature_map(x)
        conv1_out = self.conv1(feat)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)

        # Residual connection from conv1 output
        residual = self.res_proj(conv1_out)
        out = conv3_out + residual
        out = out.flatten(1)  # (B, 4*8)
        logits = self.head(out)
        return torch.sigmoid(logits)


def QCNN() -> QCNNModel:
    """Return a pre‑configured instance of :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
