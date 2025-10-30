from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

__all__ = ["HybridSelfAttention"]

class HybridSelfAttention:
    """
    Quantum hybrid self‑attention circuit.

    The circuit follows the structure of the QuantumClassifierModel
    while adding a self‑attention style entanglement block.
    It accepts external rotation and entangle parameters to emulate
    the classical attention weighting, and outputs expectation
    values of Pauli‑Z observables as logits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers after the initial encoding.
    """
    def __init__(self, n_qubits: int = 4, depth: int = 2) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        """
        Construct the parameterised circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for RX/RX/RZ on each qubit, shape (3*n_qubits,).
        entangle_params : np.ndarray
            CRX angles between adjacent qubits, shape (n_qubits-1,).
        inputs : np.ndarray
            Input features, shape (n_qubits,).

        Returns
        -------
        QuantumCircuit
            Fully constructed circuit ready for execution.
        """
        circuit = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Input encoding (simple RX)
        for i, val in enumerate(inputs):
            circuit.rx(val, i)

        # Rotation block
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entangling block (CRX)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # Variational depth
        weight_index = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                circuit.ry(weight_index, q)
                weight_index += 1
            for q in range(self.n_qubits - 1):
                circuit.cz(q, q + 1)

        # Measurement
        circuit.measure(range(self.n_qubits), range(self.n_qubits))
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int | None = None,
    ) -> np.ndarray:
        """
        Execute the circuit and return expectation values of Pauli‑Z.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation parameters for the circuit.
        entangle_params : np.ndarray
            Entanglement parameters.
        inputs : np.ndarray
            Input features.
        shots : int, optional
            Number of shots; defaults to the class attribute.

        Returns
        -------
        np.ndarray
            Expectation values for each qubit, shape (n_qubits,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        result = execute(
            circuit,
            backend=self.backend,
            shots=shots or self.shots,
            memory=True,
        ).result()
        counts = result.get_counts(circuit)
        total = sum(counts.values())

        # Pauli‑Z expectation: (-1)^bitstring
        expectations = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            value = int(bitstring, 2)
            for qubit in range(self.n_qubits):
                parity = (-1) ** ((value >> qubit) & 1)
                expectations[qubit] += parity * cnt
        expectations /= total
        return expectations
