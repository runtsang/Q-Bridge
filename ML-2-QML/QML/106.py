import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence
import numpy as np

class QuantumRBFKernel(tq.QuantumModule):
    """Quantum kernel using a parameterized encoding circuit.

    The circuit encodes data x and y into a quantum state via Ry rotations,
    then applies a trainable entangling layer. The kernel value is the fidelity
    between the two encoded states. The variational parameters are optimized
    to maximize the kernel Gram matrix rank for downstream learning tasks.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Trainable entangling parameters
        self.params = nn.Parameter(torch.randn(self.n_layers, self.n_wires, 3))

    def _encode(self, q_device: tq.QuantumDevice, data: torch.Tensor, sign: int = 1) -> None:
        """Apply data‑dependent Ry rotations; sign controls direction for x/y."""
        for i in range(self.n_wires):
            params = sign * data[:, i] if data.shape[1] > i else None
            func_name_dict["ry"](q_device, wires=[i], params=params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a single pair of vectors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        # Encode x
        self._encode(self.q_device, x, sign=1)
        # Apply trainable layers
        for layer in range(self.n_layers):
            for i in range(self.n_wires):
                func_name_dict["ry"](self.q_device, wires=[i], params=self.params[layer, i])
            for i in range(self.n_wires - 1):
                func_name_dict["cz"](self.q_device, wires=[i, i+1])
        # Reset for y
        self.q_device.reset_states(x.shape[0])
        self._encode(self.q_device, y, sign=-1)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix for lists of tensors."""
        device = a[0].device
        k = torch.zeros(len(a), len(b), device=device)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                k[i, j] = self.forward(xi, yj)
        return k.detach().cpu().numpy()

__all__ = ["QuantumRBFKernel"]
