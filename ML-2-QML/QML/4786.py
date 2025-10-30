"""
Quantum self‑attention module using Qiskit and a sampler.

The circuit implements a parameterised rotation for each query/key pair
followed by controlled‑X entanglement.  A state‑vector sampler produces
probabilities that are used as attention weights.  The interface
mirrors the original SelfAttention() factory.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

__all__ = ["SelfAttention"]


class QuantumSelfAttention:
    """Qiskit implementation of a self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used to encode a single attention head.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("aer_simulator_statevector")
        self.sampler = Sampler(self.backend)

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """Build a circuit that maps rotation_params and entangle_params
        to a probability distribution over n_qubits.
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotate each qubit according to the provided angles
        for i in range(self.n_qubits):
            circuit.ry(rotation_params[3 * i], i)
            circuit.rz(rotation_params[3 * i + 1], i)
            circuit.rx(rotation_params[3 * i + 2], i)

        # Entangle adjacent qubits
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)

        # Apply entangle angles as controlled‑RY
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return a probability distribution per sample.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation gates (shape (n_qubits, 3)).
        entangle_params : np.ndarray
            Parameters for the controlled‑X gates (shape (n_qubits-1,)).
        inputs : np.ndarray
            Not used by the circuit but kept for API compatibility.
        shots : int
            Number of execution shots for the sampler.

        Returns
        -------
        np.ndarray
            Probability distribution of shape (batch, n_qubits).
        """
        batch_size = inputs.shape[0]
        probs = np.zeros((batch_size, self.n_qubits))

        for idx in range(batch_size):
            circ = self._build_circuit(rotation_params[idx], entangle_params[idx])
            job = self.sampler.run(circ, shots=shots)
            result = job.result()
            # Convert counts to probabilities
            counts = result.get_counts(circ)
            prob = np.zeros(self.n_qubits)
            for state, count in counts.items():
                # Interpret state string as binary index
                idx_bin = int(state, 2)
                prob[idx_bin] = count / shots
            probs[idx] = prob

        return probs


def SelfAttention():
    """Factory returning a quantum self‑attention instance."""
    return QuantumSelfAttention(n_qubits=4)
