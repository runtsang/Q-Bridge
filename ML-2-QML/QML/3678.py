"""Hybrid quantum convolutional filter and classifier module."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, List

class ConvGen:
    """
    Composite quantum filter + variational classifier that mimics the classical
    interface while exploiting quantum parallelism.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, depth: int = 3, shots: int = 1024) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.num_qubits = kernel_size ** 2
        self.shots = shots
        # Use the qasm simulator for measurement‑based observables
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit, self.encodings, self.weights, self.observables = self._build_classifier_circuit(self.num_qubits, depth)

    def _build_classifier_circuit(self, num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a layered variational ansatz with explicit encoding and variational parameters.
        Returns the circuit, encoding list, weight list, and observables.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        # Encoding layer will be added per input in run()
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, [encoding], [weights], observables

    def _encode_data(self, circuit: QuantumCircuit, data: np.ndarray) -> QuantumCircuit:
        """
        Append an encoding layer that sets a rotation on each qubit according
        to whether the corresponding datum exceeds the threshold.
        """
        for idx, val in enumerate(data.flatten()):
            angle = np.pi if val > self.threshold else 0.0
            circuit.rx(angle, idx)
        return circuit

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the composite filter+classifier on the supplied data.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            Array of per‑qubit probabilities of measuring |1>, i.e. a 2‑D vector.
        """
        circ = QuantumCircuit(self.num_qubits)
        circ = self._encode_data(circ, data)
        circ.append(self.circuit, range(self.num_qubits))
        circ.measure_all()

        job = qiskit.execute(
            circ,
            backend=self.backend,
            shots=self.shots,
        )
        result = job.result()
        counts = result.get_counts(circ)
        total = sum(counts.values())
        probs = np.zeros(self.num_qubits)
        for bitstring, cnt in counts.items():
            for i, bit in enumerate(bitstring[::-1]):  # LSB is qubit 0
                if bit == "1":
                    probs[i] += cnt
        probs /= total
        return probs

__all__ = ["ConvGen"]
