"""Hybrid classical model with optional quantum-like modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridQuantumNAT"]

class HybridQuantumNAT(nn.Module):
    """Classical implementation of the hybrid architecture.  It mirrors the
    quantum variant but replaces all quantum sub‑modules with their
    classical counterparts, enabling rapid prototyping and benchmarking
    on classical hardware.
    """

    # --------------------------------------------------------------------------- #
    #  Classical CNN‑FC backbone
    # --------------------------------------------------------------------------- #
    class _CNNBackbone(nn.Module):
        def __init__(self, in_channels: int = 1) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.features(x)
            flat = feat.view(x.shape[0], -1)
            out = self.fc(flat)
            return self.norm(out)

    # --------------------------------------------------------------------------- #
    #  Classical encoder (identity)
    # --------------------------------------------------------------------------- #
    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    # --------------------------------------------------------------------------- #
    #  Classical feed‑forward head
    # --------------------------------------------------------------------------- #
    class _QFFHead(nn.Module):
        def __init__(self, in_features: int = 4, hidden_dim: int = 32, out_features: int = 2) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear2(F.relu(self.linear1(x)))

    # --------------------------------------------------------------------------- #
    #  Simple linear classifier
    # --------------------------------------------------------------------------- #
    class _Classifier(nn.Module):
        def __init__(self, in_features: int, num_classes: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    def __init__(self, num_classes: int = 2, use_qff: bool = True) -> None:
        super().__init__()
        self.backbone = self._CNNBackbone()
        self.encoder = self._Encoder()
        self.qff_head = self._QFFHead() if use_qff else nn.Identity()
        self.classifier = self._Classifier(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # (batch, 4)
        enc_feat = self.encoder(feat)    # (batch, 4)
        if isinstance(self.qff_head, nn.Identity):
            out = self.classifier(enc_feat)
        else:
            out = self.classifier(self.qff_head(enc_feat))
        return out
