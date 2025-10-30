"""Hybrid quantum autoencoder integrating a quantum NAT encoder, a sampler‑based decoder,
and a quantum kernel for similarity regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler

# Import quantum NAT model (reference 4)
from QuantumNAT import QFCModel

# Import quantum kernel (reference 3)
from QuantumKernelMethod import Kernel


class HybridAutoencoder(tq.QuantumModule):
    """A quantum autoencoder that uses a quantum NAT encoder, a RealAmplitudes decoder
    sampled with Qiskit, and a quantum kernel for regularisation.
    """
    def __init__(self, latent_dim: int = 4, num_trash: int = 0, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.encoder = QFCModel()
        self.latent_dim = latent_dim
        self.num_trash = num_trash

        # Build a parameterised ansatz for decoding
        self.decoder_ansatz = RealAmplitudes(latent_dim + num_trash, reps=1)
        self.decoder_params = list(self.decoder_ansatz.parameters)
        self.sampler = Sampler()

        # Quantum kernel for similarity
        self.kernel = Kernel()
        self.kernel.gamma = kernel_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (bsz, 1, 28, 28) for MNIST‑style data.
        Returns:
            recon: reconstruction probabilities of shape (bsz, 2) from the sampler.
            sim: similarity score between latent and input via the quantum kernel.
        """
        # Encode with quantum NAT
        latent = self.encoder(x)  # shape (bsz, 4)
        bsz = latent.size(0)

        # Prepare a list of circuits with bound parameters
        circuits = []
        for i in range(bsz):
            qc = QuantumCircuit(self.latent_dim + self.num_trash)
            # Bind latent values to the parameters of the ansatz
            param_bindings = {p: float(latent[i, j]) for j, p in enumerate(self.decoder_params)}
            qc = self.decoder_ansatz.bind_parameters(param_bindings)
            circuits.append(qc)

        # Sample probabilities for each circuit
        result = self.sampler.run(circuits)
        probs = torch.tensor(result.probabilities, device=x.device, dtype=torch.float32)

        # Kernel similarity between latent and original input
        flat_x = x.view(bsz, -1)
        sims = []
        for i in range(bsz):
            sims.append(self.kernel(latent[i], flat_x[i]))
        sim = torch.stack(sims)

        return probs, sim

    def kernel_similarity(self, latent: torch.Tensor, flat_x: torch.Tensor) -> torch.Tensor:
        """Compute kernel similarity between latent and input."""
        return self.kernel(latent, flat_x)


__all__ = ["HybridAutoencoder"]
