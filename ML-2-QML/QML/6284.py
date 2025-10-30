"""Quantum‑centric kernel that combines a variational kernel with a
Qiskit self‑attention block.  The implementation follows the
structure of the original ``Kernel`` module but replaces the
classical RBF with a quantum kernel evaluated on a fixed ansatz.
The attention block is built with Qiskit and returns measurement
counts, which are converted into a simple feature vector that
pre‑processes the data before the kernel evaluation.

The public API matches the classical helper so that the hybrid
experiments can be swapped in without changing downstream code.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumKernelAnsatz(tq.QuantumModule):
    """Fixed ansatz that encodes two data vectors in opposite directions."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that returns the overlap of two encoded states."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumSelfAttention:
    """Qiskit self‑attention block that produces a measurement‑based
    feature vector.  The counts are converted into a flat array that
    can be fed into the quantum kernel."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts(circuit)
        # Convert counts dict to a fixed‑length vector
        vec = np.zeros(2 ** self.n_qubits, dtype=float)
        for state, cnt in counts.items():
            idx = int(state[::-1], 2)  # reverse bit order to match Qiskit convention
            vec[idx] = cnt
        # Normalise to probability distribution
        return vec / self.shots

class QuantumKernelAttention:
    """Hybrid quantum kernel that optionally preprocesses data with a
    Qiskit self‑attention block before evaluating the overlap kernel."""
    def __init__(self, n_wires: int = 4, attention_type: str = "quantum") -> None:
        self.kernel = QuantumKernel(n_wires)
        self.attention_type = attention_type
        if attention_type == "quantum":
            self.attention = QuantumSelfAttention(n_qubits=n_wires)
        else:
            self.attention = None

    def forward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        if self.attention is not None:
            if rotation_params is None or entangle_params is None:
                raise ValueError("Rotation and entangle params required for quantum attention.")
            # Pre‑process each vector with the attention circuit
            x = self.attention.run(rotation_params, entangle_params)
            y = self.attention.run(rotation_params, entangle_params)
        # Convert to torch tensors for the kernel module
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return self.kernel(x_t, y_t)

    def kernel_matrix(
        self,
        a: Sequence[np.ndarray],
        b: Sequence[np.ndarray],
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.array(
            [
                [
                    self.forward(x, y, rotation_params, entangle_params).item()
                    for y in b
                ]
                for x in a
            ]
        )

def kernel_matrix(
    a: Sequence[np.ndarray],
    b: Sequence[np.ndarray],
    attention_type: str = "quantum",
    rotation_params: np.ndarray | None = None,
    entangle_params: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience wrapper mirroring the classical API."""
    qa = QuantumKernelAttention(attention_type=attention_type)
    return qa.kernel_matrix(a, b, rotation_params, entangle_params)


__all__ = ["QuantumKernelAttention", "kernel_matrix"]
