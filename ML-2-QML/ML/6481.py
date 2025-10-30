"""Enhanced Quanvolution hybrid model.

Combines classical convolutional feature extraction with a fully connected head
inspired by Quantum‑NAT.  The filter consists of two convolutional blocks
followed by ReLU and max‑pooling, mirroring the architecture of QFCModel.
The classifier uses a linear projection to 10 classes, followed by
BatchNorm1d and log_softmax.  This design preserves the simplicity of the
original Quanvolution while adding depth and regularisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """Classical hybrid model that merges classical quanvolution filter
    with a QFCModel‑style fully‑connected head.

    The filter performs two conv‑ReLU‑pool stages, producing a tensor of
    shape (batch, 16, 7, 7).  The features are flattened and passed
    through a linear head to 10 logits.  A BatchNorm1d is applied to
    the logits before log‑softmax.
    """
    def __init__(self) -> None:
        super().__init__()
        # filter
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )
        self.norm = nn.BatchNorm1d(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        logits = self.fc(flat)
        logits = self.norm(logits)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
