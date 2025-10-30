"""Hybrid quantum kernel + self‑attention module.

This module mirrors the classical implementation but replaces the RBF kernel
with a fixed TorchQuantum ansatz and the self‑attention block with a
Qiskit‑based circuit that interprets kernel entries as rotation angles.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

class KernalAnsatz(tq.QuantumModule):
    """Fixed ansatz of single‑qubit ry rotations."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.func_list = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluating a fixed TorchQuantum ansatz."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumSelfAttention:
    """Qiskit circuit implementing a self‑attention style block."""

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> float:
        """Return probability of measuring the all‑zero bitstring."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts(circuit)
        return counts.get("0" * self.n_qubits, 0) / self.shots

class HybridQuantumKernelAttention:
    """Hybrid kernel‑attention module for the quantum back‑end.

    The forward pass computes a quantum kernel matrix using the fixed
    TorchQuantum ansatz and then applies a quantum self‑attention circuit
    that reinterprets each kernel row as rotation angles.
    """

    def __init__(self, n_wires: int = 4, embed_dim: int = 4) -> None:
        self.kernel = Kernel(n_wires)
        self.attention = QuantumSelfAttention(n_qubits=n_wires)
        self.embed_dim = embed_dim

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel matrix."""
        mat = []
        for x in a:
            row = []
            for y in b:
                val = self.kernel(x, y)
                row.append(val)
            mat.append(torch.stack(row))
        return torch.stack(mat)

    def attention_matrix(self, kernel: torch.Tensor) -> torch.Tensor:
        """Apply quantum self‑attention over the kernel matrix."""
        n, m = kernel.shape
        out = torch.zeros_like(kernel)
        for i in range(n):
            # Use the i‑th row of the kernel as rotation parameters.
            # Pad or truncate to 12 parameters (4 qubits × 3 angles).
            params = kernel[i].cpu().numpy()
            if params.size < 12:
                params = np.pad(params, (0, 12 - params.size), "constant")
            else:
                params = params[:12]
            entangle = np.zeros(self.attention.n_qubits - 1)
            weight = self.attention.run(params, entangle)
            out[i] = kernel[i] * weight
        return out

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Compute hybrid representation for ``x`` given reference set ``ref``."""
        k = self.kernel_matrix(x, ref)
        return self.attention_matrix(k)

__all__ = ["HybridQuantumKernelAttention"]
