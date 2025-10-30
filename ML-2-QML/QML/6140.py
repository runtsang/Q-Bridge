"""Quantum auto‑encoder implementation using Qiskit."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler

class QuantumAutoencoder:
    """A simple quantum auto‑encoder that maps a latent vector to a quantum state."""
    def __init__(self, n_qubits: int, reps: int = 3) -> None:
        self.n_qubits = n_qubits
        self.reps = reps
        self.sampler = StatevectorSampler()
        self.ansatz = RealAmplitudes(n_qubits, reps=reps)

    def _build_circuit(self, latent: np.ndarray) -> QuantumCircuit:
        """
        Construct a quantum circuit that encodes the latent vector into a state.
        The latent vector is assumed to have length equal to n_qubits.
        """
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        for i, angle in enumerate(latent):
            circuit.rx(angle, qr[i])
        circuit.compose(self.ansatz, qr, inplace=True)
        return circuit

    def encode(self, latent: np.ndarray) -> np.ndarray:
        """
        Encode a classical latent vector into a quantum state and return the statevector.
        """
        circuit = self._build_circuit(latent)
        result = self.sampler.run(circuit).result()
        state = result.get_statevector(circuit)
        return state

    def decode(self, state: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Decode a quantum statevector back to a classical vector.
        For simplicity, we return the amplitude magnitudes as the decoded vector.
        """
        return np.abs(state[:n_bits])

    def run(self, latent: np.ndarray) -> np.ndarray:
        """
        Full encode‑decode pipeline: encode the latent, then decode back.
        """
        state = self.encode(latent)
        decoded = self.decode(state, self.n_qubits)
        return decoded
