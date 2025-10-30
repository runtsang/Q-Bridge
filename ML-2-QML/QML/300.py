"""Quantum self‑attention block using Qiskit.

The class builds a parameterized circuit that maps a set of rotation parameters to a probability distribution
over the qubits. These probabilities are interpreted as attention weights.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

class SelfAttentionDual:
    """
    Quantum implementation of a self‑attention block.

    The circuit is parameterized by `rotation_params` and `entangle_params`. After execution on a
    simulator the measurement results are converted into a probability vector that can be used as
    attention weights. The class exposes a `run` method that returns a NumPy array of shape
    (seq_len,) containing the weights.
    """

    def __init__(self, n_qubits: int = 4, backend=None):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits used in the circuit.
        backend : qiskit.providers.Backend, optional
            Quantum backend. If None, uses AerSimulator('qasm_simulator').
        """
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Construct a parameterized circuit for a single batch element.
        """
        qc = QuantumCircuit(self.qr, self.cr)
        # Apply Ry rotations with the provided parameters
        for i in range(self.n_qubits):
            qc.ry(rotation_params[i], i)
        # Entangle neighboring qubits
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        # Optional entanglement parameterised by entangle_params
        for i, p in enumerate(entangle_params):
            qc.rz(p, i % self.n_qubits)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit on the backend and return a probability distribution over qubits.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (n_qubits,) containing rotation angles for Ry gates.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) containing angles for Rz gates used as entanglement knobs.
        shots : int
            Number of measurement shots.

        Returns
        -------
        probs : np.ndarray
            Array of shape (n_qubits,) with the probability of measuring |1> on each qubit.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts(qc)
        # Convert counts to probabilities for each qubit
        probs = np.zeros(self.n_qubits)
        for state, count in counts.items():
            for i, bit in enumerate(reversed(state)):
                if bit == '1':
                    probs[i] += count
        probs /= shots
        return probs
