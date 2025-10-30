"""QuantumRegression module – quantum implementation.

This file recreates the original QML seed but adds a hybrid quantum‑kernel
branch and a pure MLP branch for comparison.  The public class
`QuantumRegression` can be instantiated with ``mode`` equal to ``'quantum'``,
``'kernel'`` or ``'ml'``.  In the quantum mode a variational circuit is
executed on a simulated device.  In the kernel mode the same fixed
ansatz used in the seed is evaluated to produce a similarity vector,
which is then passed to a classical linear head.  The ml mode simply
runs a small MLP on the input features.

All components are built on top of TorchQuantum so that the module can
be used with both CPU and GPU devices.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Optional

# --------------------------------------------------------------------------- #
# 1. Data generation – quantum version from the QML seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create superposition states |ψ> = cosθ|0…0> + e^{iφ} sinθ|1…1>."""
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

# --------------------------------------------------------------------------- #
# 2. Dataset – returns complex states
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """Dataset that emits quantum state vectors."""
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
# 3. Variational sub‑module – from the QML seed
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Randomised layer followed by trainable single‑qubit rotations."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

# --------------------------------------------------------------------------- #
# 4. Quantum kernel – identical to the ML side but implemented as a
#    pure TorchQuantum module
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Fixed ansatz that evaluates a similarity between two real vectors."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires

    def _encode(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        for i, wire in enumerate(range(self.n_wires)):
            tq.RY(params=x[i], wires=wire)(q_device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires)
        qdev.reset_states(1)
        self._encode(qdev, x)
        state_x = qdev.states.clone()
        qdev.reset_states(1)
        self._encode(qdev, y)
        state_y = qdev.states.clone()
        overlap = torch.abs(torch.sum(state_x.conj() * state_y))
        return overlap

# --------------------------------------------------------------------------- #
# 5. Hybrid quantum model – supports quantum, kernel, or classical mode
# --------------------------------------------------------------------------- #
class HybridQuantumModel(tq.QuantumModule):
    """Quantum‑variational or kernel‑based regression head."""
    def __init__(
        self,
        mode: str,
        num_wires: int = 4,
        support_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.n_wires = num_wires

        if self.mode == "quantum":
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
            self.q_layer = QLayer(num_wires)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.head = nn.Linear(num_wires, 1)
        elif self.mode == "kernel":
            assert support_vectors is not None, "support_vectors required for kernel mode"
            self.kernel = QuantumKernel(n_wires)
            self.support_vectors = support_vectors
            self.head = nn.Linear(len(support_vectors), 1)
        elif self.mode == "ml":
            # Plain MLP head for direct regression on real vectors
            self.head = nn.Sequential(
                nn.Linear(num_wires, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        if self.mode == "quantum":
            features = self.measure(qdev)
            return self.head(features).squeeze(-1)
        else:
            raise RuntimeError("HybridQuantumModel.forward expects a quantum device only in quantum mode")

# --------------------------------------------------------------------------- #
# 6. Public wrapper – matches the original API
# --------------------------------------------------------------------------- #
class QuantumRegression(tq.QuantumModule):
    """
    Main entry point.  The constructor accepts the same arguments as
    the classical wrapper but the underlying implementation switches
    between a variational circuit, a kernel head, or a simple MLP.
    """
    def __init__(
        self,
        mode: str = "quantum",
        num_wires: int = 4,
        support_vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "quantum":
            self.model = HybridQuantumModel(mode="quantum", num_wires=num_wires)
        elif self.mode == "kernel":
            self.model = HybridQuantumModel(mode="kernel", num_wires=num_wires, support_vectors=support_vectors)
        elif self.mode == "ml":
            self.model = HybridQuantumModel(mode="ml", num_wires=num_wires)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.mode == "quantum":
            bsz = batch.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.model.n_wires, bsz=bsz, device=batch.device)
            self.model.encoder(qdev, batch)
            self.model.q_layer(qdev)
            return self.model(qdev)
        elif self.mode == "kernel":
            support = self.model.support_vectors.to(batch.device)
            kvec = torch.stack(
                [self.model.kernel(b, s) for s in support]
            ).transpose(0, 1)  # shape: (batch, support)
            return self.model.head(kvec)
        elif self.mode == "ml":
            return self.model.head(batch).squeeze(-1)
        else:
            raise RuntimeError("Unreachable")

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
