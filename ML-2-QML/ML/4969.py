"""Hybrid fraud detection pipeline combining photonic-inspired layers, self‑attention, and a regression‑classifier stack.

The class is built on top of the original ``FraudDetection`` and ``SelfAttention`` modules,
and extends the architecture with a lightweight estimator and a final classifier.
It is fully classical, relying only on PyTorch, NumPy and the seed modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import numpy as np
from torch import nn

# Seed components
from FraudDetection import FraudLayerParameters
from SelfAttention import SelfAttention
from EstimatorQNN import EstimatorQNN
from QuantumClassifierModel import build_classifier_circuit

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection pipeline that combines photonic‑inspired layers,
    a self‑attention block, an estimator regressor and a final classifier.
    """

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        attention_params: np.ndarray,
        estimator: bool = True,
    ) -> None:
        super().__init__()

        # Fraud‑detection core (2→2 layers)
        self.fraud_core = _layer_from_params(fraud_params, clip=False)

        # Self‑attention block
        self.attention = SelfAttention()

        # Estimator (regressor)
        self.estimator = EstimatorQNN() if estimator else None

        # Classifier (2→2)
        self.classifier, _, _, _ = build_classifier_circuit(num_features=2, depth=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch, 2).
        """
        # 1. Fraud‑detection core
        fraud_out = self.fraud_core(x)

        # 2. Self‑attention (requires numpy arrays)
        attn_out_np = self.attention.run(
            rotation_params=self.attention_params,
            entangle_params=np.zeros_like(self.attention_params),
            inputs=fraud_out.detach().cpu().numpy(),
        )
        attn_out = torch.from_numpy(attn_out_np).to(x.device)

        # 3. Estimation
        if self.estimator is not None:
            est_out = self.estimator(attn_out)
            est_feat = torch.cat([est_out, est_out], dim=1)
        else:
            est_feat = attn_out

        # 4. Classification
        logits = self.classifier(est_feat)
        return logits


__all__ = ["FraudDetectionHybrid"]
