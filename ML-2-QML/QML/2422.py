"""Hybrid quantum kernel: quantum autoencoder ansatz with swap‑test overlap."""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence
import torchquantum as tq
from torchquantum.functional import func_name_dict

class HybridKernel(tq.QuantumModule):
    """Quantum kernel that emulates an autoencoder circuit and measures state overlap."""
    def __init__(self,
                 latent_dim: int = 3,
                 trash_dim: int = 2,
                 reps: int = 5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = latent_dim + 2 * trash_dim + 1  # auxiliary qubit
        self.q_device = tq.QuantumDevice(n_wires=self.num_qubits)

        # Simple data‑encoding ansatz: Ry rotations on the first latent+trash qubits
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(latent_dim + trash_dim)
        ]

        # Auxiliary qubit for the swap test
        self.aux_wire = self.num_qubits - 1

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset device for batch size
        self.q_device.reset_states(x.shape[0])

        # Encode first data point with positive parameters
        for info in self.ansatz:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Swap‑test preparation
        self.q_device.h(self.aux_wire)
        for i in range(self.trash_dim):
            tq.cswap(self.q_device,
                     wires=[self.aux_wire,
                            self.latent_dim + i,
                            self.latent_dim + self.trash_dim + i])

        self.q_device.h(self.aux_wire)

        # Encode second data point with negative parameters (phase flip)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap of the two encoded states."""
        self.forward(x, y)
        # The first basis state amplitude gives the overlap
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  latent_dim: int = 3,
                  trash_dim: int = 2) -> np.ndarray:
    """Compute the Gram matrix using the quantum autoencoder kernel."""
    kernel = HybridKernel(latent_dim, trash_dim)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
