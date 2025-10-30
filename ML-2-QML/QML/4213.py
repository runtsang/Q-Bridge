import numpy as np
import qiskit
from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

class QuantumAutoencoder:
    """Quantum auto‑encoder using a RealAmplitudes variational circuit for encoding
    and another for decoding. Latent information is extracted via a sampler
    and fed into the decoder circuit."""
    def __init__(self, input_dim: int, latent_dim: int = 3, reps: int = 2, shots: int = 1024):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.sampler = Sampler()
        self.encoder = RealAmplitudes(num_qubits=latent_dim, reps=reps)
        self.decoder = RealAmplitudes(num_qubits=latent_dim, reps=reps)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        # Map the classical input vector to parameters of the encoder circuit
        theta = inputs.reshape(-1)
        circ = self.encoder.assign_parameters(dict(zip(self.encoder.parameters, theta)), inplace=False)
        result = self.sampler.run(circ, shots=self.shots).result()
        expectations = result.get_expectation_value('Z', circ)
        return expectations

    def decode(self, latent: np.ndarray) -> np.ndarray:
        # Use the latent vector as parameters for the decoder circuit
        theta = latent.reshape(-1)
        circ = self.decoder.assign_parameters(dict(zip(self.decoder.parameters, theta)), inplace=False)
        result = self.sampler.run(circ, shots=self.shots).result()
        expectations = result.get_expectation_value('Z', circ)
        return expectations

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        latent = self.encode(inputs)
        recon = self.decode(latent)
        return recon

def QuantumHybridAutoencoder(input_dim: int,
                             latent_dim: int = 3,
                             reps: int = 2,
                             shots: int = 1024) -> QuantumAutoencoder:
    return QuantumAutoencoder(input_dim, latent_dim, reps, shots)

__all__ = ["QuantumAutoencoder", "QuantumHybridAutoencoder"]
