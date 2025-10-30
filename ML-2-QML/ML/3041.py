"""Hybrid fraud‑detection model with classical and quantum‑inspired layers.

This module defines ``FraudDetectionHybrid`` – a PyTorch model that:
1. Builds a classical analogue of the photonic circuit described in the original seed.
2. Adds a configurable QLSTM layer that can be either a classical LSTM or a quantum‑enhanced LSTM.
3. Exposes a single forward pass that feeds through the photonic‑inspired layers, the optional LSTM, and a final classifier.

The design follows the “combination” scaling paradigm: the classical backbone is extended by quantum‑inspired or quantum‑realised components, but the overall API remains the same as the original ``FraudDetection`` model, making it drop‑in compatible for downstream pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1.  Classical photonic‑inspired layer
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single 2‑mode photonic layer – identical to the seed."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(val: float, bound: float) -> float:
    return max(-bound, min(bound, val))

def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> nn.Module:
    # 1‑to‑1 mapping of the seed’s linear and bias terms
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a sequential model that mirrors the seed’s photonic circuit."""
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2.  Quantum‑enhanced LSTM
# --------------------------------------------------------------------------- #
# Import the quantum LSTM implementation from the QML module.
# The relative import path may need adjustment depending on package layout.
from.qml_module import QLSTM  # type: ignore[import]

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model that fuses photonic‑inspired layers with a
    quantum‑enhanced LSTM (or a classical LSTM if no qubits are requested)."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        hidden_dim: int = 64,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.photonic = build_fraud_detection_program(input_params, layers)
        self.input_to_hidden = nn.Linear(2, hidden_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                batch_first=False,
            )
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch, 2) representing a sequence of
            2‑dimensional transaction features.

        Returns
        -------
        torch.Tensor
            Fraud‑risk logits of shape (seq_len, batch, 1).
        """
        seq_len, batch, _ = x.shape
        # Flatten the temporal dimension for the photonic layers
        x = x.view(seq_len * batch, 2)
        x = self.photonic(x)
        x = self.input_to_hidden(x)
        # Restore the temporal structure
        x = x.view(seq_len, batch, -1)
        # Process the sequence with the chosen LSTM implementation
        outputs, _ = self.lstm(x)
        logits = self.output_head(outputs)
        return logits

__all__ = ["FraudDetectionHybrid"]
