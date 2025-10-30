from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumAutoencoder:
    """
    Variational quantum autoencoder that maps a classical latent vector
    to a quantum state and back using a RealAmplitudes ansatz.
    """
    def __init__(self, latent_dim: int, reps: int = 3, shots: int = 1024) -> None:
        self.latent_dim = latent_dim
        self.shots = shots
        self.backend = Sampler()
        self.encoder_ansatz = RealAmplitudes(latent_dim, reps=reps)
        self.decoder_ansatz = RealAmplitudes(latent_dim, reps=reps)

    def _build_circuit(self, data: np.ndarray, ansatz: RealAmplitudes) -> QuantumCircuit:
        """
        Builds a circuit that first embeds the classical vector via RX rotations
        and then applies the variational ansatz.
        """
        qc = QuantumCircuit(self.latent_dim)
        for i, val in enumerate(data):
            qc.rx(val, i)
        qc.append(ansatz.to_instruction(), range(self.latent_dim))
        return qc

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode a classical vector into the quantum latent space.
        Returns the statevector as a numpy array.
        """
        data = np.asarray(data).flatten()
        if data.size!= self.latent_dim:
            raise ValueError(f"Input vector must be of length {self.latent_dim}")
        circ = self._build_circuit(data, self.encoder_ansatz)
        result = self.backend.run(circ, shots=self.shots).result()
        return np.real_if_close(result.get_statevector(), tol=1e-6)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode a quantum latent vector back to a classical representation.
        """
        latent = np.asarray(latent).flatten()
        if latent.size!= self.latent_dim:
            raise ValueError(f"Latent vector must be of length {self.latent_dim}")
        circ = self._build_circuit(latent, self.decoder_ansatz)
        result = self.backend.run(circ, shots=self.shots).result()
        return np.real_if_close(result.get_statevector(), tol=1e-6)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Full autoencoding pipeline: encode and then decode.
        """
        latent = self.encode(data)
        return self.decode(latent)


# ------------------------------------------------------------
# SamplerQNN wrapper for seamless integration with PyTorch
# ------------------------------------------------------------

def QuantumAutoencoderQNN(latent_dim: int, reps: int = 3) -> SamplerQNN:
    """
    Returns a SamplerQNN that implements the variational autoencoder
    as a differentiable layer.  The QNN can be inserted into a
    PyTorch model as an nn.Module.
    """
    encoder_ansatz = RealAmplitudes(latent_dim, reps=reps)
    decoder_ansatz = RealAmplitudes(latent_dim, reps=reps)

    def circuit(x):
        qc = QuantumCircuit(latent_dim)
        for i, val in enumerate(x):
            qc.rx(val, i)
        qc.append(encoder_ansatz, range(latent_dim))
        qc.barrier()
        qc.append(decoder_ansatz, range(latent_dim))
        return qc

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=list(encoder_ansatz.parameters) + list(decoder_ansatz.parameters),
        interpret=lambda x: x,
        output_shape=(latent_dim,),
        sampler=Sampler(),
    )
    return qnn


__all__ = [
    "QuantumAutoencoder",
    "QuantumAutoencoderQNN",
]
