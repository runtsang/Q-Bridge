"""Hybrid quantum kernel method that chains a quanvolution filter with a
TorchQuantum ansatz.  The public API mirrors the classical version.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import qiskit
from qiskit.circuit.random import random_circuit
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import op_name_dict

__all__ = ["Conv", "KernalAnsatz", "HybridKernel", "kernel_matrix"]


# --------------------------------------------------------------------------- #
#  Quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Filter circuit used for quanvolution layers."""

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the quantum circuit on classical data and return per‑qubit
        probabilities of measuring ``|1>``.
        """
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Count occurrences of each qubit being |1>
        counts = np.zeros(self.n_qubits, dtype=float)
        for key, val in result.items():
            for i, bit in enumerate(reversed(key)):
                if bit == "1":
                    counts[i] += val
        probs = counts / (self.shots * len(result))
        return probs


def Conv(kernel_size: int = 2, shots: int = 100, threshold: float = 127):
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return QuanvCircuit(kernel_size, backend, shots, threshold)


# --------------------------------------------------------------------------- #
#  Quantum RBF kernel (fixed ansatz)
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

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


# --------------------------------------------------------------------------- #
#  Hybrid quantum kernel
# --------------------------------------------------------------------------- #
class HybridKernel(tq.QuantumModule):
    """Hybrid quantum kernel that first applies a quanvolution filter
    and then evaluates a fixed TorchQuantum ansatz on the resulting
    probability vectors.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        super().__init__()
        self.filter = Conv(kernel_size=kernel_size, shots=shots, threshold=threshold)
        self.n_wires = kernel_size ** 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid quantum kernel.

        Parameters
        ----------
        x, y
            Tensors of shape ``(batch, features)`` where ``features`` equals
            ``kernel_size ** 2``.  Each row is treated as a 2‑D image.
        """
        # Apply quanvolution filter to each sample
        x_feat = torch.tensor(
            [
                self.filter.run(
                    xi.cpu().numpy().reshape(-1, self.n_wires)
                )
                for xi in x
            ],
            dtype=torch.float32,
        )
        y_feat = torch.tensor(
            [
                self.filter.run(
                    yi.cpu().numpy().reshape(-1, self.n_wires)
                )
                for yi in y
            ],
            dtype=torch.float32,
        )

        # Evaluate the fixed ansatz
        self.ansatz(self.q_device, x_feat, y_feat)
        return torch.abs(self.q_device.states.view(-1)[0])


# --------------------------------------------------------------------------- #
#  Utility to compute a Gram matrix
# --------------------------------------------------------------------------- #
def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two sequences of tensors."""
    kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])
