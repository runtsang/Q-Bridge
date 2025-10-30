"""Hybrid classical self‑attention + fraud‑detection model.

The module combines the SelfAttention helper and the FraudDetection network
from the provided seed projects, exposing a single class that can be used
directly in downstream pipelines.  The class accepts the same parameter
types as the seeds but stitches them together into a coherent forward
pass.  The design mirrors the quantum counterpart, enabling side‑by‑side
experimentation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Re‑use the original parameter container
from dataclasses import dataclass
from typing import Iterable

# Import the seed helpers (paths are relative to this file)
from.SelfAttention import SelfAttention
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program


@dataclass
class HybridParameters:
    """Container for the parameters of the hybrid model."""
    # Self‑attention weights
    rotation_params: np.ndarray
    entangle_params: np.ndarray
    # Fraud‑detection layer parameters
    fraud_input_params: FraudLayerParameters
    fraud_layer_params: Iterable[FraudLayerParameters]


class HybridSelfAttentionFraudDetector(nn.Module):
    """Composite classical model integrating attention and fraud detection.

    The forward pass first applies a self‑attention block to ``inputs``,
    then feeds the attention‑weighted representation into a fraud‑detection
    network.  The structure mirrors the quantum counterpart, so that
    ``run`` can be used interchangeably in classical or quantum
    experimentation.
    """

    def __init__(self, embed_dim: int, params: HybridParameters):
        super().__init__()
        # Self‑attention helper
        self.attention = SelfAttention(embed_dim=embed_dim)
        # Fraud‑detection sequential model
        self.fraud_model = build_fraud_detection_program(
            params.fraud_input_params,
            params.fraud_layer_params,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run the composite model."""
        # Self‑attention output
        attn_out = self.attention.run(
            self.rotation_params,
            self.entangle_params,
            inputs.numpy(),
        )
        attn_tensor = torch.as_tensor(attn_out, dtype=torch.float32)
        # Fraud‑detection prediction
        return self.fraud_model(attn_tensor)

    # Helper to expose parameters for consistency with the seed
    @property
    def rotation_params(self) -> np.ndarray:
        return self.attention.rotation_params

    @property
    def entangle_params(self) -> np.ndarray:
        return self.attention.entangle_params
