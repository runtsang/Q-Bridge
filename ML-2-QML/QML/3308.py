from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

# Quantum convolution (adapted from QML Conv.py)
class QuanvCircuit:
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """
        Execute the quantum filter on a 2‑D array.

        Args:
            data: 2‑D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        total = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
            total += val

        return counts / (total * self.n_qubits)

# Classifier builder (adapted from QML seed)
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuantumClassifierModel:
    """
    Quantum classifier that couples a data‑uploading convolutional filter
    with a parameter‑shift variational ansatz.  The public API matches the
    classical counterpart, enabling direct comparison experiments.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        conv_kernel: int = 2,
        conv_threshold: float = 127,
        shots: int = 100,
    ) -> None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel, backend, shots, conv_threshold)
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = backend

    def run(self, data: np.ndarray) -> List[float]:
        """
        Run the full quantum pipeline on a 2‑D input patch.

        Args:
            data: 2‑D array of shape (conv_kernel, conv_kernel).

        Returns:
            List[float]: Expectation value of each Z‑observable.
        """
        prob = self.conv.run(data)  # scalar in [0,1]

        # Bind the same probability to all variational parameters
        param_binds = [{theta: prob for theta in self.weights}]
        job = execute(
            self.circuit,
            self.backend,
            shots=self.circuit.num_qubits * 10,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        expectations = []
        total_counts = sum(counts.values())
        for op in self.observables:
            i = op.primitive.paulis.index("Z")
            exp = 0.0
            for bitstring, cnt in counts.items():
                eigen = 1.0 if bitstring[-(i + 1)] == "0" else -1.0
                exp += eigen * cnt
            expectations.append(exp / total_counts)
        return expectations
