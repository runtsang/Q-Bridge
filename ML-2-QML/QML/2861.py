"""Quantum implementation of a quanvolution autoencoder using Qiskit."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuanvolutionAutoencoder(nn.Module):
    """Quantum‑classical hybrid autoencoder that applies a variational circuit to 2×2 patches."""
    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 reps: int = 5,
                 device: str | None = None) -> None:
        super().__init__()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.total_qubits = num_latent + 2 * num_trash + 1  # +1 auxiliary for swap‑test
        self.reps = reps

        qr = QuantumRegister(self.total_qubits, "q")
        cr = ClassicalRegister(self.total_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Parameterized Ry gates for the 2×2 pixel patch
        patch_params = [Parameter(f'patch_{i}') for i in range(4)]
        for i, p in enumerate(patch_params):
            circuit.ry(p, qr[i])

        # Variational ansatz on the first num_latent + num_trash qubits
        ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
        circuit.compose(ansatz, list(range(num_latent + num_trash)), inplace=True)

        # Swap‑test style auxiliary qubit
        aux = qr[self.total_qubits - 1]
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)

        # Measure all qubits
        for i in range(self.total_qubits):
            circuit.measure(qr[i], cr[i])

        self.circuit = circuit

        # Sampler QNN
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=patch_params,
            weight_params=ansatz.parameters,
            interpret=lambda x: x,
            output_shape=self.total_qubits,
            sampler=self.sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to every 2×2 patch of the input image."""
        bsz, c, h, w = x.shape
        assert h == w and h % 2 == 0, "Input must be a square image with even dimensions."
        outputs = []
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                patch = x[:, :, i:i+2, j:j+2]  # shape (bsz, c, 2, 2)
                patch = patch.view(bsz, -1)  # flatten to 4 or 4*c
                patch_np = patch.cpu().numpy()
                result = self.qnn(patch_np)
                outputs.append(result)

        # Concatenate outputs of all patches
        out = torch.cat([torch.tensor(p, device=x.device) for p in outputs], dim=1)
        return out

__all__ = ["QuanvolutionAutoencoder"]
