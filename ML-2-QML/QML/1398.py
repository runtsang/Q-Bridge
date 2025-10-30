import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler


class Autoencoder__gen362(nn.Module):
    """Quantum decoder for the hybrid autoencoder."""

    def __init__(self, latent_dim: int, output_dim: int, reps: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.reps = reps

        # Build the parameterized circuit
        self.qc = self._build_circuit()

        # Sampler for statevector sampling
        self.sampler = StatevectorSampler()

        # Build the QNN
        self.qnn = SamplerQNN(
            circuit=self.qc,
            input_params=[f'latent_{i}' for i in range(self.latent_dim)],
            weight_params=list(self.qc.parameters)[self.latent_dim:],
            interpret=self._interpret,
            output_shape=(self.output_dim,),
            sampler=self.sampler
        )

    def _build_circuit(self):
        """Construct a circuit with latent parameters and a RealAmplitudes ansatz."""
        qr = QuantumRegister(self.output_dim, name='q')
        cr = ClassicalRegister(1, name='c')
        qc = QuantumCircuit(qr, cr)

        # Apply RY gates with latent parameters
        for i in range(self.latent_dim):
            qc.ry(f'latent_{i}', qr[i])

        # Add RealAmplitudes ansatz
        ansatz = RealAmplitudes(self.output_dim, reps=self.reps)
        qc.append(ansatz, qr)

        # Dummy measurement; the sampler uses statevector
        qc.measure(qr[0], cr[0])

        return qc

    def _interpret(self, probabilities: np.ndarray) -> np.ndarray:
        """Interpret the probability distribution as expectation values of Pauli‑Z."""
        batch_size = probabilities.shape[0]
        exp_vals = np.zeros((batch_size, self.output_dim))
        # Precompute parity for each qubit
        basis = np.arange(2 ** self.output_dim)
        for i in range(self.output_dim):
            mask = 1 << i
            bits = ((basis & mask) > 0).astype(int)
            parity = 1 - 2 * bits  # +1 for 0, -1 for 1
            exp_vals[:, i] = np.sum(probabilities * parity, axis=1)
        return exp_vals

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the quantum decoder on a batch of latent vectors."""
        latent_np = latent.detach().cpu().numpy()
        outputs = self.qnn(latent_np)
        return torch.tensor(outputs, dtype=latent.dtype, device=latent.device)


__all__ = ["Autoencoder__gen362"]
