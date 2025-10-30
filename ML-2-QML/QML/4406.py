"""Hybrid quantum self‑attention module.

The circuit encodes the classical inputs as Ry rotations, applies
parameterized rotations and entanglement gates, and derives attention
weights from the measurement statistics.  The public ``run`` method
mirrors the classical interface: ``run(backend, rotation_params,
entangle_params, inputs, shots)``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector


class HybridSelfAttention:
    """Quantum self‑attention with parameterized rotations and CRX entanglement.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= len(inputs)).
    backend : qiskit.providers.BaseBackend | None, default None
        Quantum backend; if None a local QASM simulator is used.
    """

    def __init__(self, n_qubits: int, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        """Construct the parameterized circuit for a single data point."""
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encode inputs as Ry rotations
        for i in range(min(self.n_qubits, len(inputs))):
            circuit.ry(inputs[i], i)

        # Apply rotation parameters as RX/RY/RZ sequence
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entangle adjacent qubits with CRX gates
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # Measure all qubits
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """Execute the circuit and return a classical attention vector.

        Parameters
        ----------
        backend : qiskit.providers.BaseBackend
            Quantum backend to execute the circuit.
        rotation_params : np.ndarray
            Rotation parameters of shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Entanglement parameters of shape (n_qubits - 1,).
        inputs : np.ndarray
            Input vector of length <= n_qubits.
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Attention output of shape (1,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, backend, shots=shots)
        counts = job.result().get_counts(circuit)

        # Convert counts to probabilities
        probs = np.array(
            [
                counts.get(format(i, "0{}b".format(self.n_qubits)), 0)
                for i in range(2**self.n_qubits)
            ]
        )
        probs = probs / shots

        # Map measurement outcomes to attention weights:
        # weight_i = sum_{bitstrings} (-1)**bit_i * prob
        weights = np.zeros(self.n_qubits)
        for bitstring, p in zip(counts.keys(), probs):
            bits = np.array([int(b) for b in bitstring])
            for i in range(self.n_qubits):
                weights[i] += ((-1) ** bits[i]) * p

        # Normalize weights with softmax
        weights = np.exp(weights) / np.sum(np.exp(weights))

        # Compute weighted sum of the input vector
        output = np.dot(weights, inputs)
        return output


def SelfAttention() -> HybridSelfAttention:
    """Return a default hybrid quantum self‑attention instance."""
    return HybridSelfAttention(n_qubits=4)
