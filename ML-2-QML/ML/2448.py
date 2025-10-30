from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

# Import the photonic‑style parameters and builder from the original ML seed
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program
# Import the classical convolutional filter from Conv.py
from.Conv import Conv


class FraudDetectionModel(nn.Module):
    """
    Hybrid fraud‑detection model that combines a classical convolutional
    filter with a photonic‑inspired neural network.

    The model processes 2×2 image patches: first a learnable Conv filter
    extracts a scalar feature, which is duplicated to a 2‑dimensional vector
    and fed into a sequential network built from `FraudLayerParameters`.
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        # Classical convolutional filter
        self.conv_filter = Conv(kernel_size=conv_kernel_size, threshold=conv_threshold)
        # Photonic‑style neural network (no intermediate layers for brevity)
        self.fraud_net = build_fraud_detection_program(fraud_params, [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of 2×2 patches with shape (batch, 1, 2, 2).

        Returns
        -------
        torch.Tensor
            Fraud probability logits of shape (batch, 1).
        """
        batch = x.shape[0]
        # Convert patches to numpy for the classical Conv filter
        patches = x.view(batch, 2, 2).cpu().numpy()
        conv_feats = []
        for patch in patches:
            feat = self.conv_filter.run(patch)
            conv_feats.append(feat)
        conv_feats = torch.tensor(conv_feats, dtype=torch.float32, device=x.device)
        # Duplicate scalar feature to a 2‑dimensional vector
        features = conv_feats.unsqueeze(-1).repeat(1, 2)
        return self.fraud_net(features)


__all__ = ["FraudDetectionModel"]
