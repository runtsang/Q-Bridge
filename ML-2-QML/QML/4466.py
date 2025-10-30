"""Quantum‑centric hybrid module that unifies self‑attention, quantum kernel, and graph QNN.

The implementation uses Qiskit for the self‑attention circuit and
TorchQuantum for the kernel and graph QNN.  The class
``HybridSelfAttentionKernelQNN`` exposes a single interface that can be
instantiated with ``use_quantum=True`` (default) to run all components
on quantum hardware or simulators.  The API matches the classical
version while providing genuine quantum operations.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Quantum self‑attention (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""

    def __init__(self, n_qubits: int = 4, backend=None) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)


# --------------------------------------------------------------------------- #
#  Quantum kernel (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Quantum kernel implemented with TorchQuantum."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple encoding: Ry on each wire
        self.ry = tq.RY(has_params=True, trainable=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Encode x and y with opposite phases
        self.q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            self.ry(self.q_device, wires=i, params=x[:, i])
        for i in range(self.n_wires):
            self.ry(self.q_device, wires=i, params=-y[:, i])
        return torch.abs(self.q_device.states.view(-1)[0]) ** 2


# --------------------------------------------------------------------------- #
#  Quantum graph QNN (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuantumGraphQNN(tq.QuantumModule):
    """Quantum graph neural network built from random layers."""

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(qnn_arch)
        self.n_wires = self.arch[-1]
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Build random layers
        self.layers: List[tq.QuantumModule] = []
        for _ in range(len(self.arch) - 1):
            layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch = x.shape[0]
        self.q_device.reset_states(batch)
        # Encode input as Ry rotations
        for i in range(self.n_wires):
            tq.RY(has_params=True)(self.q_device, wires=i, params=x[:, i])
        activations: List[torch.Tensor] = [self.q_device.states.view(-1).clone()]
        for layer in self.layers:
            layer(self.q_device)
            activations.append(self.q_device.states.view(-1).clone())
        return activations


# --------------------------------------------------------------------------- #
#  Hybrid controller (quantum)
# --------------------------------------------------------------------------- #
class HybridSelfAttentionKernelQNN:
    """Hybrid quantum module that combines self‑attention, kernel, and graph QNN."""

    def __init__(
        self,
        embed_dim: int = 4,
        qnn_arch: Sequence[int] = (4, 8, 4),
        gamma: float = 1.0,
        use_quantum: bool = True,
    ) -> None:
        self.use_quantum = use_quantum
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)
        self.kernel = QuantumKernel()
        self.qnn = QuantumGraphQNN(qnn_arch)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> Tuple[dict, torch.Tensor, List[torch.Tensor]]:
        """Return (attention_counts, kernel_output, qnn_output)."""
        attn_counts = self.attention.run(rotation_params, entangle_params, shots=shots)
        # Kernel: compare each sample with itself
        dummy_input = torch.rand(1, self.kernel.n_wires)
        ker_output = self.kernel(dummy_input, dummy_input)
        qnn_output = self.qnn(dummy_input)
        return attn_counts, ker_output, qnn_output


def SelfAttention(use_quantum: bool = True, **kwargs) -> HybridSelfAttentionKernelQNN:
    """Convenient factory that mirrors the original SelfAttention interface."""
    return HybridSelfAttentionKernelQNN(use_quantum=use_quantum, **kwargs)


__all__ = [
    "HybridSelfAttentionKernelQNN",
    "SelfAttention",
]
