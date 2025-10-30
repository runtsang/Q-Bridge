"""HybridClassifier: quantum implementation using TorchQuantum."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import QuantumDevice, QuantumModule


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that encodes two inputs and returns overlap amplitude."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = QuantumDevice(n_wires=self.n_wires, bsz=1)
        # Simple encoding: Ry for each feature, followed by a random layer
        self.encoder = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        self.q_device.reset_states(1)
        # Encode x
        for idx, wire in enumerate(range(self.n_wires)):
            func_name_dict["ry"](self.q_device, wires=wire, params=x[0, idx])
        self.encoder(self.q_device)
        # Encode y with negative angles
        for idx, wire in enumerate(range(self.n_wires)):
            func_name_dict["ry"](self.q_device, wires=wire, params=-y[0, idx])
        self.encoder(self.q_device)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix between two lists of tensors."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Quantum analogue of a fully connected layer with trainable gates."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        tq.hadamard(qdev, wires=3)


class HybridClassifierQML(tq.QuantumModule):
    """Quantum classifier that optionally includes a quantum fully‑connected layer."""

    def __init__(self, n_qubits: int = 4, depth: int = 2, use_qfc: bool = False) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QuantumFullyConnectedLayer(n_wires=n_qubits) if use_qfc else None
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_qubits)
        self.depth = depth

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        # pool the input into a 1‑D vector of length n_qubits
        pooled = F.avg_pool2d(x, 6).view(bsz, self.n_qubits)
        self.encoder(qdev, pooled)
        for _ in range(self.depth):
            if self.q_layer:
                self.q_layer(qdev)
            self.measure(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = [
    "HybridClassifierQML",
    "QuantumKernel",
    "kernel_matrix",
    "QuantumFullyConnectedLayer",
]
