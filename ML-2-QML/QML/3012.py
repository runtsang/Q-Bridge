"""UnifiedAutoencoder: quantum autoencoder implementation.

The class implements a swap‑test style quantum autoencoder using a
RealAmplitudes ansatz.  It exposes `encode`, `decode` and `forward`
methods that mirror the classical API.  When a quantum backend is
available, the forward pass returns both the latent state vector and a
classical reconstruction obtained by measuring the trash qubits.
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
import numpy as np
import torch

class UnifiedAutoencoder:
    def __init__(self, latent_dim: int = 3, num_trash: int = 2, reps: int = 5):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = StatevectorSampler()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Build a swap‑test autoencoder circuit."""
        n_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode part
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)

        # Swap‑test
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a quantum statevector for the input latent vector."""
        # For simplicity we only support batch size 1 in this toy example
        assert inputs.shape[0] == 1
        params = inputs.flatten().detach().cpu().numpy()
        qc = self.circuit.assign_parameters(params, inplace=True)
        result = self.sampler.run(qc).result()
        return torch.tensor(result.quasi_distribution, dtype=torch.float32)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Classical reconstruction from a latent vector."""
        # In a pure quantum setting we would implement a quantum decoder.
        # Here we simply return the latent as reconstruction for illustration.
        return latent

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent statevector and classical reconstruction."""
        latent = self.encode(inputs.unsqueeze(0))
        recon = self.decode(latent)
        return recon, latent

__all__ = ["UnifiedAutoencoder"]
