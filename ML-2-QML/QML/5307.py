import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence

class HybridQuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the overlap between two data points
    encoded by a parameterised Ry‑CZ circuit.  The module can be
    instantiated with an arbitrary number of qubits and optionally
    accepts pre‑processed classical features.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute |⟨ψ(x)|ψ(y)⟩| for 1‑D tensors x and y.
        """
        self.q_device.reset_states(1)
        # Encode x
        for i, val in enumerate(x):
            func_name_dict["Ry"](self.q_device, wires=[i], params=val)
        for i in range(self.n_wires - 1):
            func_name_dict["CZ"](self.q_device, wires=[i, i + 1])
        # Encode -y by applying inverse gates in reverse order
        for i, val in enumerate(y):
            func_name_dict["Ry"](self.q_device, wires=[i], params=-val)
        for i in reversed(range(self.n_wires - 1)):
            func_name_dict["CZ"](self.q_device, wires=[i, i + 1])
        # Return absolute overlap with the initial state |0…0⟩
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        mat = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.forward(x, y).item()
        return mat

__all__ = ["HybridQuantumKernel"]
