"""Hybrid quantum autoencoder + quantum kernel."""
import torch
import torchquantum as tq
import numpy as np
from typing import Sequence
from torchquantum.functional import func_name_dict

class QuantumAutoencoder(tq.QuantumModule):
    """Variational quantum autoencoder that maps classical data to a latent subspace."""
    def __init__(self, num_latent: int, num_trash: int):
        super().__init__()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.n_wires = num_latent + 2 * num_trash + 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        """Simple Ry‑rotation ansatz on the first `num_latent + num_trash` qubits."""
        return [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.num_latent + self.num_trash)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class HybridAutoencoderKernel(tq.QuantumModule):
    """Combines the variational autoencoder with a quantum kernel over the latent space."""
    def __init__(self, num_latent: int, num_trash: int):
        super().__init__()
        self.autoencoder = QuantumAutoencoder(num_latent, num_trash)

    def _latent_state(self, x: torch.Tensor) -> torch.Tensor:
        q_device = tq.QuantumDevice(n_wires=self.autoencoder.n_wires)
        self.autoencoder(q_device, x)
        return q_device.states.view(-1)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Quantum kernel as absolute inner product of encoded states."""
        state_x = self._latent_state(x)
        state_y = self._latent_state(y)
        return torch.abs(torch.vdot(state_x, state_y))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Gram matrix over latent quantum states."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

__all__ = ["HybridAutoencoderKernel"]
