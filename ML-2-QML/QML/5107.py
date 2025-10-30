import numpy as np
import torch
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes

class QuantumLatentEncoder:
    """Variational circuit that maps a latent vector to a quantum state and returns
    expectation values of Pauli‑Z on each qubit, used as a quantum reconstruction."""
    def __init__(self, latent_dim: int, backend=None, shots: int = 1000):
        self.latent_dim = latent_dim
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        # Build a parameterised circuit with RealAmplitudes
        self.circuit = RealAmplitudes(latent_dim, reps=2)
        self.circuit.measure_all()

    def _expectation_z(self, counts: dict) -> np.ndarray:
        total = sum(counts.values())
        exp = np.zeros(self.latent_dim)
        for state, cnt in counts.items():
            prob = cnt / total
            bits = [int(b) for b in state[::-1]]  # little‑endian
            for i, b in enumerate(bits):
                exp[i] += prob * (1 if b == 0 else -1)
        return exp

    def run(self, latent: np.ndarray | torch.Tensor) -> np.ndarray:
        """Execute the circuit for a batch of latent vectors.
        Returns expectation of Z for each qubit."""
        if hasattr(latent, "numpy"):
            latent = latent.numpy()
        latent = np.atleast_2d(latent)
        expectations = []
        for vec in latent:
            param_dict = {p: v for p, v in zip(self.circuit.parameters, vec)}
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_dict])
            counts = job.result().get_counts()
            expectations.append(self._expectation_z(counts))
        return np.array(expectations)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.run(latent)).float()

def QuantumEncoder(latent_dim: int, shots: int = 1000) -> QuantumLatentEncoder:
    """Convenience factory mirroring the classical Autoencoder factory."""
    return QuantumLatentEncoder(latent_dim=latent_dim, shots=shots)

__all__ = ["QuantumLatentEncoder", "QuantumEncoder"]
