"""Hybrid quantum‑classical regression – quantum side.

The quantum module expands on the seed by embedding a QCNN‑style
variational circuit that reduces the number of qubits through
pooling.  The final measurement is passed to a classical linear
layer, mirroring the hybrid architecture on the classical side.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


# ------------------------------------------------------------------
# Dataset – identical to the classical seed for consistency.
# ------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data in the form of a superposition of |0…0> and |1…1>."""
    rng = np.random.default_rng(12345)
    thetas = rng.uniform(0, 2 * np.pi, size=samples)
    phis = rng.uniform(0, 2 * np.pi, size=samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * np.eye(2 ** num_wires)[0] \
                    + np.exp(1j * phis[i]) * np.sin(thetas[i]) * np.eye(2 ** num_wires)[-1]

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a tensor of complex states and a scalar target."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# ------------------------------------------------------------------
# QCNN‑style quantum layers.
# ------------------------------------------------------------------
class QConvLayer(tq.QuantumModule):
    """A single convolutional block – rotations + entanglement."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.cx = tq.CX()

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            self.cx(qdev, control=wire, target=wire + 1)


class QPoolLayer(tq.QuantumModule):
    """Pooling that discards the last half of the qubits."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        full = self.measure(qdev)
        return full[:, : self.n_wires // 2]


class QCNNQuantumModule(tq.QuantumModule):
    """Full QCNN circuit composed of alternating conv and pool layers."""

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        current_wires = num_wires
        for _ in range(depth):
            self.layers.append(QConvLayer(current_wires))
            self.layers.append(QPoolLayer(current_wires))
            current_wires //= 2  # Pooling halves the qubit count

        self.final_measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, QConvLayer):
                layer(qdev)
            elif isinstance(layer, QPoolLayer):
                layer(qdev)
        return self.final_measure(qdev)


# ------------------------------------------------------------------
# Quantum regression model – QCNN + classical head.
# ------------------------------------------------------------------
class QModel(tq.QuantumModule):
    """End‑to‑end quantum regression model."""

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.qcnn = QCNNQuantumModule(num_wires, depth=depth)
        # After depth layers the wire count is num_wires // (2**depth)
        self.head = nn.Linear(num_wires // (2 ** depth), 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=bsz,
                                device=state_batch.device)
        # Encode the raw complex amplitudes directly into the device
        qdev.set_state(state_batch)
        q_features = self.qcnn(qdev)
        return self.head(q_features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
