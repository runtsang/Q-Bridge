"""Hybrid self‑attention: quantum implementation using Qiskit.

The module defines a SelfAttentionHybrid class that implements a
self‑attention block where the attention weights are derived from
a small parameterised quantum circuit.  The circuit encodes the
input token embeddings as rotation angles on each qubit, applies
a variational entangling layer, and measures the qubits to obtain
a probability distribution over the tokens.  The class is fully
compatible with the original API: it exposes a `run` method that
takes rotation and entanglement parameters and returns the
measurement counts.

The public factory function `SelfAttention()` returns an instance
configured for 4 qubits, matching the original seed.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

class SelfAttentionHybrid:
    """Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used to encode each token.
    backend : qiskit.providers.Backend, optional
        Quantum backend to execute the circuit.  Defaults to the
        Aer qasm simulator.
    shots : int, default 1024
        Number of shots for each execution.
    """

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """Construct a parameterised circuit for a single token.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened array of rotation angles (3 * n_qubits).
        entangle_params : np.ndarray
            Array of entanglement angles (n_qubits - 1).

        Returns
        -------
        QuantumCircuit
            The constructed circuit ready for execution.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Apply rotations to each qubit
        for i in range(self.n_qubits):
            rx_angle = rotation_params[3 * i]
            ry_angle = rotation_params[3 * i + 1]
            rz_angle = rotation_params[3 * i + 2]
            circuit.rx(rx_angle, i)
            circuit.ry(ry_angle, i)
            circuit.rz(rz_angle, i)

        # Entangle neighbouring qubits with controlled‑RZ gates
        for i in range(self.n_qubits - 1):
            circuit.crz(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> dict:
        """Execute the circuit and return measurement counts.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened rotation angles for all qubits.
        entangle_params : np.ndarray
            Entanglement angles for the C‑RZ gates.

        Returns
        -------
        dict
            Dictionary of bit‑string counts produced by the simulator.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=self.shots)
        return job.result().get_counts(circuit)

def SelfAttention() -> SelfAttentionHybrid:
    """Factory returning a default instance matching the original seed."""
    return SelfAttentionHybrid(n_qubits=4)

__all__ = ["SelfAttentionHybrid", "SelfAttention"]
