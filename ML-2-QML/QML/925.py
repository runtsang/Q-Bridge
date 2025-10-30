"""Quantum regression head and hybrid model.

This module implements a quantum submodule that can be attached to a
classical feature extractor.  The head is a variational circuit that
takes a classical feature vector as input via a classical encoder
and outputs a scalar prediction.  The class ``QuantumHead`` is
designed to be dropped into the hybrid model defined in the
``ml`` module.

The module also re‑exports the data utilities for completeness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

# ----------------------------------------------------------------------
# Data utilities (same as ML)
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a synthetic regression dataset where the target depends on
    a nonlinear combination of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Compatibility wrapper around the original dataset."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Quantum head
# ----------------------------------------------------------------------
class QuantumHead(tq.QuantumModule):
    """
    Variational quantum circuit that maps a classical feature vector to
    a scalar output.  The circuit uses a random layer followed by
    trainable single‑qubit rotations and a measurement of Pauli‑Z.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires

        # Encoder that maps a classical vector of length 2**num_wires to a state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )

        # Variational part
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

        # Measurement and classical head
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass expects a batch of classical feature vectors of
        shape (bsz, 2**num_wires).  They are encoded into a quantum state,
        processed by the variational circuit, and finally projected to a scalar.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumHead", "RegressionDataset", "generate_superposition_data"]
