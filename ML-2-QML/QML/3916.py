from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit import execute
from typing import Iterable, Tuple

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuanvCircuit:
    """Filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """Run the quantum circuit on classical data."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

class HybridConvClassifierQ:
    """Quantum hybrid classifier that chains a quanvolution filter with a variational ansatz."""
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 kernel_size: int = 2,
                 conv_threshold: float = 127,
                 shots: int = 100):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quanvolution pre‑processing
        self.conv = QuanvCircuit(kernel_size, self.backend, shots, conv_threshold)

        # Variational classifier
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def run(self, data):
        """Run data through the quanvolution filter and variational classifier."""
        # Pre‑processing
        conv_prob = self.conv.run(data)

        # Bind encoding parameters to the probability output
        param_binds = {param: conv_prob for param in self.encoding}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result().get_counts(self.circuit)

        # Compute probability of measuring |1> for each qubit
        probs = []
        for i in range(self.num_qubits):
            count_one = sum(val for key, val in result.items() if key[-(i+1)] == '1')
            probs.append(count_one / (self.shots * len(result)))
        return probs

    def predict(self, data):
        probs = self.run(data)
        return np.argmax(probs)

__all__ = ["HybridConvClassifierQ"]
