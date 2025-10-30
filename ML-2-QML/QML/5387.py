"""Quantum regression module with optional quantum transformer encoder.

This module extends the original QuantumRegression seed by adding:
- A flexible data generator that can produce complex amplitude arrays.
- A quantum encoder that uses a random layer followed by parameterized rotations.
- An optional quantum transformer block that replaces the classical encoder
  with a multi‑head attention module implemented in TorchQuantum.
- A classical linear head that maps the measurement results to a scalar.

The public API is identical to the original seed: RegressionDataset,
QModel, and generate_superposition_data are still available for
backward compatibility.  New convenience classes are also exported.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int,
                                samples: int,
                                *,
                                use_complex: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a tuple (states, labels).  The states are complex amplitude
    arrays of shape (samples, 2**num_wires).  Labels are computed as
    sin(2θ) * cos(φ) where θ and φ are random angles used to generate
    the states.
    """
    if not use_complex:
        raise ValueError("Quantum data must be complex.")
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary with ``states`` and ``target``.
    ``states`` are complex amplitude tensors.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum encoder
# --------------------------------------------------------------------------- #
class QuantumEncoder(tq.QuantumModule):
    """
    Encodes a batch of amplitude vectors into a quantum device using a
    random layer followed by trainable RX/RY rotations.
    """
    def __init__(self, num_wires: int, n_ops: int = 30):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

# --------------------------------------------------------------------------- #
# Optional quantum transformer block
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(tq.QuantumModule):
    """
    A lightweight transformer block implemented with TorchQuantum.
    It applies a multi‑head attention layer followed by a feed‑forward
    network where the projections are realised by small quantum circuits.
    """
    class QAttention(tq.QuantumModule):
        def __init__(self, n_wires: int, n_heads: int):
            super().__init__()
            self.n_wires = n_wires
            self.n_heads = n_heads
            # Simple encoder mapping each head to a distinct subset of wires
            self.encoders = nn.ModuleList([
                tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]}]
                ) for i in range(n_heads)
            ])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: QuantumDevice) -> torch.Tensor:
            out = []
            for enc in self.encoders:
                enc(qdev)
                out.append(self.measure(qdev))
            return torch.stack(out, dim=1).mean(dim=1)

    class QFeedForward(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.enc = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: QuantumDevice) -> torch.Tensor:
            self.enc(qdev)
            return self.measure(qdev)

    def __init__(self, n_wires: int, n_heads: int = 4, n_ffn_wires: int = 8):
        super().__init__()
        self.attn = self.QAttention(n_wires, n_heads)
        self.ffn = self.QFeedForward(n_ffn_wires)

    def forward(self, qdev: QuantumDevice) -> torch.Tensor:
        attn_out = self.attn(qdev)
        ffn_out = self.ffn(qdev)
        return attn_out + ffn_out

# --------------------------------------------------------------------------- #
# Quantum regression model
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """
    Quantum regression model that uses a quantum encoder, an optional
    quantum transformer block, and a classical linear head.
    """
    def __init__(self,
                 num_wires: int,
                 *,
                 use_transformer: bool = False,
                 transformer_cfg: dict | None = None):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.encoder_layer = QuantumEncoder(num_wires)
        self.transformer = None
        if use_transformer:
            cfg = transformer_cfg or {}
            self.transformer = QuantumTransformerBlock(
                n_wires=cfg.get("n_wires", num_wires),
                n_heads=cfg.get("n_heads", 4),
                n_ffn_wires=cfg.get("n_ffn_wires", 8),
            )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.encoder_layer(qdev)
        if self.transformer is not None:
            self.transformer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Backward‑compatibility aliases
# --------------------------------------------------------------------------- #
HybridQuantumRegressionModel = QModel

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QModel",
    "HybridQuantumRegressionModel",
]
