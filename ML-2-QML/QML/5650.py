import torch
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class UnifiedAutoencoder:
    """Quantum refinement module for latent vectors."""
    def __init__(self, latent_dim: int, reps: int = 3):
        self.latent_dim = latent_dim
        self.reps = reps
        self.theta = ParameterVector("theta", self.latent_dim)
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.theta,
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=latent_dim,
            sampler=Sampler(),
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Builds a variational circuit with RealAmplitudes ansatz."""
        qr = QuantumRegister(self.latent_dim, "q")
        qc = QuantumCircuit(qr)

        # Encode input angles as parameterized RY gates
        for i in range(self.latent_dim):
            qc.ry(self.theta[i], qr[i])

        # Ansatz
        ansatz = RealAmplitudes(self.latent_dim, reps=self.reps)
        qc.compose(ansatz, range(self.latent_dim), inplace=True)

        # Measure each qubit
        cr = ClassicalRegister(self.latent_dim, "c")
        qc.add_register(cr)
        qc.measure(qr, cr)

        return qc

    def _interpret(self, outcome: np.ndarray) -> torch.Tensor:
        """
        Convert measurement bitstring to a float vector in [-1, 1]^latent_dim.
        Each bit is mapped: 0 -> -1, 1 -> 1.
        """
        vec = torch.tensor([1.0 if b else -1.0 for b in outcome], dtype=torch.float32)
        return vec

    def refine(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Refine a batch of latent vectors using the quantum circuit.
        Parameters
        ----------
        latents : torch.Tensor, shape (batch, latent_dim)
            Classical latent vectors to be refined.
        Returns
        -------
        torch.Tensor
            Refined latent vectors, shape (batch, latent_dim).
        """
        batch = latents.detach().cpu().numpy()
        # Wrap latent values to [0, 2π] as angles
        inputs = (batch % (2 * np.pi)).tolist()
        outputs = self.qnn.forward(inputs)
        return torch.tensor(outputs, dtype=torch.float32)

__all__ = ["UnifiedAutoencoder"]
