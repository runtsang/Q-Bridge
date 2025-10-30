"""Quantum decoder for AutoencoderGen215.

Implements a parameter‑tuned RealAmplitudes‑style circuit that takes a latent vector
as rotation angles and returns a probability distribution over a binary string
of length equal to the latent dimension. The decoder can be called from the
classical training loop to provide the reconstruction term.
"""

from __future__ import annotations

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

class QuantumDecoder:
    def __init__(self, latent_dim: int, reps: int = 3) -> None:
        self.latent_dim = latent_dim
        self.reps = reps
        self.params = ParameterVector("theta", latent_dim)
        self.circuit = QuantumCircuit(latent_dim)
        # Simple ansatz: RY(theta) on each qubit followed by CX layers
        for _ in range(reps):
            for q in range(latent_dim):
                self.circuit.ry(self.params[q], q)
            for q in range(latent_dim - 1):
                self.circuit.cx(q, q + 1)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Return probabilities for each sample in the batch.

        Args:
            z: Tensor of shape (batch, latent_dim) containing rotation angles.

        Returns:
            Tensor of shape (batch, 2**latent_dim) with probabilities.
        """
        batch_size = z.shape[0]
        probs = []
        for i in range(batch_size):
            param_dict = {self.params[j]: float(z[i, j].item()) for j in range(self.latent_dim)}
            circ = self.circuit.bind_parameters(param_dict)
            sv = Statevector(circ)
            probs.append(sv.probabilities())
        return torch.tensor(probs, dtype=torch.float32)

def build_quantum_decoder(latent_dim: int, reps: int = 3) -> QuantumDecoder:
    """Convenience factory mirroring the classical Autoencoder factory."""
    return QuantumDecoder(latent_dim=latent_dim, reps=reps)

__all__ = ["QuantumDecoder", "build_quantum_decoder"]
