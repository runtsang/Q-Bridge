"""Quantum decoder for the hybrid autoencoder."""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

class HybridAutoencoderQuantumDecoder:
    """Quantum decoder that maps a latent vector to a reconstructed data vector."""
    def __init__(
        self,
        num_latent: int,
        num_output: int,
        reps: int = 3,
        shots: int = 1024,
    ) -> None:
        self.num_latent = num_latent
        self.num_output = num_output
        self.reps = reps
        self.shots = shots
        self.circuit = self._build_circuit()
        self.num_params = len(self.circuit.parameters)

    def _build_circuit(self) -> QuantumCircuit:
        """Build a RealAmplitudes ansatz over all qubits (latent + output)."""
        total_qubits = self.num_latent + self.num_output
        qr = QuantumRegister(total_qubits, "q")
        circuit = QuantumCircuit(qr)
        circuit.compose(
            RealAmplitudes(total_qubits, reps=self.reps),
            qr,
            inplace=True,
        )
        return circuit

    def _parameterize(self, latent: np.ndarray) -> np.ndarray:
        """Map a latent vector to the circuit parameters."""
        if len(latent)!= self.num_latent:
            raise ValueError(f"Expected latent vector of length {self.num_latent}, got {len(latent)}")
        # Simple linear mapping: normalize and pad remaining parameters with zeros
        params = np.zeros(self.num_params, dtype=np.float64)
        params[:self.num_latent] = latent / (np.linalg.norm(latent) + 1e-8)
        return params

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Return a reconstructed vector from a latent vector."""
        params = self._parameterize(latent)
        state = Statevector(
            self.circuit,
            parameter_binds=[{p: val for p, val in zip(self.circuit.parameters, params)}]
        )
        probs = state.probabilities()
        expectations: List[float] = []
        for i in range(self.num_output):
            exp = 0.0
            for idx, prob in enumerate(probs):
                bit = (idx >> (self.num_latent + i)) & 1
                exp += ((-1) ** bit) * prob
            expectations.append(exp)
        return np.array(expectations)

    def run(self, latents: Iterable[np.ndarray]) -> np.ndarray:
        """Batch infer over multiple latent vectors."""
        return np.vstack([self.decode(l) for l in latents])

def create_quantum_decoder(
    num_latent: int = 3,
    num_output: int = 2,
    reps: int = 3,
    shots: int = 1024,
) -> HybridAutoencoderQuantumDecoder:
    """Factory for the quantum decoder."""
    return HybridAutoencoderQuantumDecoder(num_latent, num_output, reps, shots)

__all__ = ["HybridAutoencoderQuantumDecoder", "create_quantum_decoder"]
