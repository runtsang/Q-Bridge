"""Quantum hybrid kernel using TorchQuantum."""

from __future__ import annotations

from typing import Sequence

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class HybridKernel(tq.QuantumModule):
    """Quantum hybrid kernel that multiplies a Gaussian RBF with a quantum feature
    map implemented by a parameterized circuit.

    The depth parameter controls the number of qubits and the number of layers
    in the ansatz.  The circuit consists of a layer of Ry rotations for each
    qubit followed by a chain of CNOTs that entangle the qubits.  The
    ansatz is applied with the first data vector, then reversed with the
    second data vector (with negated parameters) to obtain the overlap.
    """

    def __init__(self, gamma: float = 1.0, depth: int = 2) -> None:
        super().__init__()
        self.gamma = gamma
        self.depth = depth
        self.n_wires = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self) -> list[dict]:
        """Build a parameterized circuit for the feature map."""
        ops = []
        # Layer of Ry rotations
        for i in range(self.n_wires):
            ops.append(
                {
                    "input_idx": [i],
                    "func": "ry",
                    "wires": [i],
                }
            )
        # Entangling layer of CNOTs (chain)
        for i in range(self.n_wires - 1):
            ops.append(
                {
                    "input_idx": [],
                    "func": "cx",
                    "wires": [i, i + 1],
                }
            )
        return ops

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the hybrid kernel value for two 1‑D vectors."""
        # Classical RBF part
        diff = x - y
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff))

        # Quantum part
        self.q_device.reset_states(1)
        # Apply ansatz with x
        for info in self.ansatz:
            params = (
                x[info["input_idx"]]
                if info["func"]!= "cx" and tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # Apply reversed ansatz with -y
        for info in reversed(self.ansatz):
            params = (
                -y[info["input_idx"]]
                if info["func"]!= "cx" and tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        q_overlap = torch.abs(self.q_device.states.view(-1)[0])
        return rbf * q_overlap

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0, depth: int = 2) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = HybridKernel(gamma=gamma, depth=depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
