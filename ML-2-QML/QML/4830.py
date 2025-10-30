"""Hybrid quantum model extending the original Quantum‑NAT architecture.

The model uses a classical CNN feature extractor followed by a quantum encoder
and a variational layer.  It is fully differentiable via torchquantum and
can be trained end‑to‑end.  The same dataset helpers from the classical
version are provided for convenient comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule
from torchquantum.functional import hadamard, sx, cnot

# --------------------------------------------------------------------------- #
# Dataset utilities (identical to the classical version)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states used for regression.

    Returns a tuple of shape (samples, 2**num_wires) and a scalar target.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset of quantum states and regression targets."""

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
# Main quantum model
# --------------------------------------------------------------------------- #
class HybridNATModel(QuantumModule):
    """Quantum‑enabled version of the HybridNATModel.

    The model consists of:
    1. Classical CNN feature extractor (identical to the classical
       implementation).
    2. General encoder that maps the flattened input into a quantum state.
    3. Variational quantum layer (`QLayer`) containing random gates and
       parameterised rotations.
    4. Measurement of all qubits followed by a linear head.
    """

    class QLayer(QuantumModule):
        """Variational layer used after the encoder."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(n_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            self.crx(qdev, wires=[0, 1])

    def __init__(self, n_wires: int = 4, num_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires

        # Classical CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Quantum encoder
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{n_wires}xRy"]
        )

        # Quantum variational layer
        self.q_layer = self.QLayer(n_wires)

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head
        self.head = nn.Linear(n_wires, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)

        # Prepare a state vector of shape (bsz, 2**n_wires)
        # We use the first 2**n_wires entries from the flattened tensor
        state = flat[:, :2 ** self.n_wires]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, state)
        self.q_layer(qdev)
        q_out = self.measure(qdev)
        out = self.head(q_out)
        return self.norm(out)


__all__ = [
    "HybridNATModel",
    "RegressionDataset",
    "generate_superposition_data",
]
