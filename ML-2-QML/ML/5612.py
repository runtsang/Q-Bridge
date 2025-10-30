"""Hybrid regression model with a classical backbone and a quantum head.

The module is designed to be drop‑in compatible with the original
``QuantumRegression`` seed but adds:
* A deep classical encoder that learns hierarchical features.
* A configurable quantum block (random layer + RX/RZ) that can be
  replaced by a full variational circuit.
* A trainable linear head that maps the quantum measurement outcomes
  to a scalar regression target.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation – keeps the original superposition signal but allows
# optional feature augmentation.
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset where each sample is a superposition
    of |0⟩ and |1⟩ states with a phase term.  The returned labels are
    a smooth function of the angles.  If ``augment`` is True the
    feature vector is concatenated with a random noise vector,
    providing a richer classical input space.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)

    if augment:
        noise = np.random.normal(0.0, 0.2, size=(samples, num_features))
        x = np.concatenate([x, noise], axis=1)

    return x, y.astype(np.float32)


# --------------------------------------------------------------------------- #
# PyTorch Dataset – mirrors the original API but accepts the new augment flag.
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Dataset that yields a dictionary with ``states`` and ``target``.
    ``states`` are float tensors ready for the classical encoder.
    """
    def __init__(self, samples: int, num_features: int, *, augment: bool = False):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, augment=augment
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Classical backbone – a multi‑layer perceptron that can be tuned
# for arbitrary depth.
# --------------------------------------------------------------------------- #
class ClassicalEncoder(nn.Module):
    """
    A flexible MLP that can be stacked with the quantum head.
    """
    def __init__(self, in_features: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, hidden_dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Quantum head – borrowed from the seed but wrapped into a reusable
# QuantumModule that can be swapped for a deeper ansatz.
# --------------------------------------------------------------------------- #
class QuantumHead(tq.QuantumModule):
    """
    Variational circuit that performs:
    * Random layer (30 ops)
    * RX/RZ rotation per wire
    * Optional entangling block (CZ or CNOT)
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, n_wires: int, entangle: bool = True):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.layer = self._QLayer(n_wires)
        self.entangle = entangle
        if entangle:
            self.cnot = tq.CNOT
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.layer(qdev)
        if self.entangle:
            for w in range(self.n_wires - 1):
                self.cnot(qdev, wires=[w, w + 1])
            self.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# Hybrid model – classical encoder + quantum head + linear readout.
# --------------------------------------------------------------------------- #
class QuantumRegressionHybrid(nn.Module):
    """
    Combines a classical network with a quantum module.  The
    forward pass is:
    1. Classical encoder transforms raw features.
    2. Classical features are encoded into the quantum device.
    3. Quantum circuit processes the states and measures.
    4. Linear head maps measurement vector to the target.
    """
    def __init__(self,
                 num_features: int,
                 hidden_dims: list[int],
                 n_qubits: int,
                 entangle: bool = True,
                 device: str | torch.device = "cpu",
                 ) -> None:
        super().__init__()
        self.classical = ClassicalEncoder(num_features, hidden_dims)
        self.quantum = QuantumHead(n_qubits, entangle=entangle)
        self.head = nn.Linear(n_qubits, 1)
        self.device = torch.device(device)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Batch of shape (B, D) where D is the feature dimension.
        Returns
        -------
        batch‑wise regression output.
        """
        # 1. Classical encoding
        encoded = self.classical(states)

        # 2. Prepare quantum device
        bsz = encoded.size(0)
        qdev = tq.QuantumDevice(n_wires=self.quantum.n_wires,
                                bsz=bsz,
                                device=self.device)

        # 3. Encode classical features into amplitudes
        self.quantum.encoder(qdev, encoded)

        # 4. Quantum evolution and measurement
        out = self.quantum(qdev)

        # 5. Readout
        return self.head(out).squeeze(-1)


__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
