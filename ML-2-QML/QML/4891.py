"""Hybrid quantum kernel‑classifier with attention ansatz and variational classifier."""

from __future__ import annotations

from typing import Tuple, Iterable
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1. Attention‑style variational ansatz
# --------------------------------------------------------------------------- #
class AttentionAnsatz(tq.QuantumModule):
    """
    Encodes classical data and applies a trainable attention‑like layer
    using cross‑qubit rotations.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Parameterized Ry gates for encoding
        self.rys = [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
        # Cross‑qubit rotations acting as entanglement
        self.crxs = [tq.CRX(has_params=True, trainable=True) for _ in range(n_wires - 1)]

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        # Encode data
        for idx, ry in enumerate(self.rys):
            ry(qdev, wires=idx, params=x[:, idx] if x.ndim > 1 else x)
        # Attention‑like entanglement
        for idx, crx in enumerate(self.crxs):
            crx(qdev, wires=[idx, idx + 1])

# --------------------------------------------------------------------------- #
# 2. Quantum kernel module
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """
    Computes a quantum kernel via a fixed circuit followed by inner‑product
    measurement. The circuit uses the AttentionAnsatz for encoding.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.attn = AttentionAnsatz(self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns the absolute overlap of the two encoded states.
        """
        self.q_device.reset_states(x.shape[0])
        # Encode x
        self.attn(self.q_device, x)
        # Unitary for y (inverse encoding)
        self.attn(self.q_device, -y)
        # Measure the first wire as a proxy for overlap
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Tuple[torch.Tensor,...],
                          b: Tuple[torch.Tensor,...]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 3. Variational quantum classifier circuit
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[tq.QuantumModule, Iterable, Iterable, Tuple[tq.Pauli,...]]:
    """
    Constructs a layered ansatz:
      • Parameterized Ry gates for each qubit
      • Depth‑wise entangling CXs
      • Final measurement observables (PauliZ on each qubit)
    """
    class QClassifier(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.rys = [tq.RY(has_params=True, trainable=True) for _ in range(num_qubits)]
            self.cxs = [tq.CX(has_params=True, trainable=True) for _ in range(depth * (num_qubits - 1))]

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for ry in self.rys:
                ry(qdev)
            idx = 0
            for _ in range(depth):
                for _ in range(num_qubits - 1):
                    self.cxs[idx](qdev, wires=[idx, idx + 1])
                    idx += 1

    classifier = QClassifier()
    observables = tuple(tq.PauliZ for _ in range(num_qubits))
    return classifier, tuple(), tuple(), observables

# --------------------------------------------------------------------------- #
# 4. Hybrid quantum model
# --------------------------------------------------------------------------- #
class HybridKernelClassifier(tq.QuantumModule):
    """
    Quantum counterpart to the classical HybridKernelClassifier.
    It:
      1) Encodes input data via the AttentionAnsatz
      2) Evaluates a quantum kernel (inner‑product) between data points
      3) Passes the kernel vector through a variational classifier
    """
    def __init__(self,
                 n_qubits: int = 4,
                 depth: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.kernel = QuantumKernel(n_wires=n_qubits)
        self.classifier, _, _, self.observables = build_classifier_circuit(n_qubits, depth)
        self.norm = nn.BatchNorm1d(n_qubits)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, n_qubits)
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                bsz=bsz,
                                device=x.device,
                                record_op=True)
        # Encode input
        self.kernel.attn(qdev, x)
        # Classifier ansatz
        self.classifier(qdev)
        # Measurement
        out = tqf.measure_all(qdev, self.observables, static=self.static_mode)
        return self.norm(out)

__all__ = ["HybridKernelClassifier", "QuantumKernel", "quantum_kernel_matrix",
           "build_classifier_circuit", "AttentionAnsatz"]
