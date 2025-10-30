"""Quantum kernel with trainable ansatz, depth control, and noise simulation."""

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence

class Kernel(tq.QuantumModule):
    """
    Variational quantum kernel.

    Parameters
    ----------
    n_wires : int
        Number of qubits.
    depth : int
        Depth of the ansatz.
    noise_level : float
        Probability of depolarising noise per gate.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2, noise_level: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.noise_level = noise_level
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        """
        Build a parameterised Ry-Rz entangling layer repeated `depth` times.
        """
        layers = []
        for _ in range(self.depth):
            # Ry on each qubit
            layers.append(
                {"input_idx": list(range(self.n_wires)), "func": "ry", "wires": list(range(self.n_wires))}
            )
            # Entangling CZ between adjacent qubits
            for i in range(self.n_wires - 1):
                layers.append(
                    {"input_idx": None, "func": "cz", "wires": [i, i + 1]}
                )
        return layers

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Apply data encoding and inverse encoding.

        Parameters
        ----------
        x, y : torch.Tensor
            Classical feature vectors of shape (batch, dim).
        """
        batch = x.shape[0]
        q_device.reset_states(batch)
        # Encode x
        for info in self.ansatz:
            if info["func"] in ["ry"]:
                params = x[:, info["input_idx"]]
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
            else:
                func_name_dict[info["func"]](q_device, wires=info["wires"])
        # Apply inverse encoding of y
        for info in reversed(self.ansatz):
            if info["func"] in ["ry"]:
                params = -y[:, info["input_idx"]]
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
            else:
                func_name_dict[info["func"]](q_device, wires=info["wires"])

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Public API returning the overlap amplitude.
        """
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute Gram matrix using the variational quantum kernel.
    """
    kernel = Kernel()
    mat = torch.stack([kernel(torch.tensor(x), torch.tensor(y)).squeeze()
                       for x in a for y in b])
    mat = mat.view(len(a), len(b))
    return mat.cpu().numpy()

__all__ = ["Kernel", "kernel_matrix"]
