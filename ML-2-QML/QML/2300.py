"""Quantum implementation of a variational latent layer for a hybrid autoencoder."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector


class HybridAutoencoder:
    """Quantum circuit implementing a variational latent layer.

    The circuit takes `latent_dim` input angles (classical parameters) and
    returns expectation values of Pauli‑Z on each qubit.  These values are
    intended to be fed into a classical decoder.
    """

    def __init__(self, latent_dim: int = 32, reps: int = 3):
        self.latent_dim = latent_dim
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim)
        qc = QuantumCircuit(qr)
        # Encode the input angles as Ry rotations
        for i in range(self.latent_dim):
            qc.ry(qiskit.circuit.Parameter(f"theta_{i}"), qr[i])
        # Variational ansatz
        qc.append(RealAmplitudes(self.latent_dim, reps=self.reps), qr)
        # Measure all qubits
        qc.measure_all()
        return qc

    def evaluate(self, angles: np.ndarray) -> np.ndarray:
        """Return expectation values of Pauli‑Z for each qubit."""
        if angles.shape[0]!= self.latent_dim:
            raise ValueError("angles must match latent_dim")
        param_binds = [{f"theta_{i}": ang for i, ang in enumerate(angles)}]
        result = self.sampler.run(self.circuit, param_binds=param_binds).result()
        state = Statevector(result.get_statevector())
        expectation = np.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            pauli = "I" * i + "Z" + "I" * (self.latent_dim - i - 1)
            expectation[i] = state.expectation_value(pauli)
        return expectation

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Convenience wrapper that accepts a list of angles."""
        return self.evaluate(np.asarray(thetas))


__all__ = ["HybridAutoencoder"]
