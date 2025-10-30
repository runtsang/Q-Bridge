from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Import the photonic fraud‑detection utilities from the ML seed
from FraudDetection import FraudLayerParameters, build_fraud_detection_program
# Import the autoencoder factory from the ML seed
from Autoencoder import Autoencoder

class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler that:
    1) Encodes the raw input with a lightweight autoencoder.
    2) Feeds the latent representation into a fraud‑detection style feed‑forward stack
       built from the supplied FraudLayerParameters.
    3) Projects onto a 2‑dimensional output and applies a softmax.
    """
    def __init__(self,
                 input_dim: int,
                 fraud_params: List[FraudLayerParameters],
                 latent_dim: int = 3) -> None:
        super().__init__()
        if len(fraud_params) < 2:
            raise ValueError("At least two FraudLayerParameters required (input + at least one hidden).")
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        # Build a sequential module mirroring the photonic circuit
        self.fraud_module = build_fraud_detection_program(fraud_params[0], fraud_params[1:])
        self.output = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Encode, fraud‑detect, then classify
        latent = self.autoencoder.encode(x)
        fraud_out = self.fraud_module(latent)
        logits = self.output(fraud_out)
        return F.softmax(logits, dim=-1)

def HybridSamplerQNN_factory(input_dim: int,
                             fraud_params: List[FraudLayerParameters],
                             latent_dim: int = 3) -> HybridSamplerQNN:
    """
    Factory that returns a fully‑configured HybridSamplerQNN instance.
    """
    return HybridSamplerQNN(input_dim, fraud_params, latent_dim)

__all__ = ["HybridSamplerQNN", "HybridSamplerQNN_factory"]
