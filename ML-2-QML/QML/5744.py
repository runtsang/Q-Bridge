from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int):
    """
    Reimplementation of the reference quantum classifier factory.
    Returns a circuit, encoding parameters, variational parameters, and observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

class QuanvClassifier:
    """
    Quantum analogue of ConvFilter: quanvolution layer followed by a variational classifier.
    The run() method returns a classification probability, and run_classification()
    returns a two‑element probability vector.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 2,
        shots: int = 100,
        threshold: float = 0.5,
        backend=None
    ):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Quanvolution filter
        self.conv_circuit = self._build_quanv_circuit(kernel_size)

        # Variational classifier
        self.classifier_circuit, self.encoding_params, self.weight_params, self.observables = build_classifier_circuit(
            self.n_qubits, depth
        )

        # Assemble full circuit
        self.circuit = self._assemble_circuit()

    def _build_quanv_circuit(self, kernel_size: int):
        n = self.n_qubits
        qc = QuantumCircuit(n)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n, 2)
        return qc

    def _assemble_circuit(self):
        qc = self.conv_circuit.copy()

        # Encode classical data
        for i, param in enumerate(self.encoding_params):
            qc.rx(param, i)

        # Apply classifier ansatz
        for i, param in enumerate(self.weight_params):
            qc.ry(param, i % self.n_qubits)
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)

        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the full circuit on a single kernel patch and return the
        average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.encoding_params[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        counts = 0
        total = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
            total += val

        avg = counts / (self.shots * self.n_qubits * total)
        return avg

    def run_classification(self, data: np.ndarray):
        """
        Return a two‑element probability vector for class 0 and class 1.
        """
        prob = self.run(data)
        return [prob, 1 - prob]

def Conv() -> QuanvClassifier:
    """
    Factory function compatible with the original Conv.py interface.
    """
    return QuanvClassifier()

__all__ = ["Conv"]
