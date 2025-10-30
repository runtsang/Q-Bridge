import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two classical vectors
    via a swap‑test circuit.  It is used both as a stand‑alone kernel
    and as the core of a variational quantum autoencoder.
    """
    def __init__(self, reps: int = 5):
        super().__init__()
        self.reps = reps

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset device to a fresh state for each batch
        q_device.reset_states(x.shape[0])

        # Encode x into the first register
        for i in range(x.shape[1]):
            func_name_dict["ry"](q_device, wires=[i], params=x[:, i])

        # Apply swap‑test with y in the second register
        for i in range(y.shape[1]):
            func_name_dict["ry"](q_device, wires=[i], params=-y[:, i])

class QuantumAutoEncoder(tq.QuantumModule):
    """Variational quantum autoencoder that maps a classical input to
    a low‑dimensional latent vector using a parameterised circuit and
    expectation‑value readout.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Quantum device and ansatz
        self.n_wires = input_dim
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.layers.RealAmplitudes(n_wires=self.n_wires, reps=2)

        # Trainable parameters for the ansatz
        self.weight_params = nn.Parameter(torch.randn(self.n_wires * 2))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return a latent vector of size (batch, latent_dim)."""
        # Encode the input data
        self.ansatz(self.q_device, x)

        # Readout expectation values of Pauli‑Z on the first latent_dim qubits
        latent = torch.stack(
            [
                self.q_device.expectation(circuit=tq.layers.PauliZ(), wires=[i])
                for i in range(self.latent_dim)
            ],
            dim=-1,
        )
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors
    using the swap‑test based quantum kernel.
    """
    kernel = KernalAnsatz()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "QuantumAutoEncoder", "kernel_matrix"]
