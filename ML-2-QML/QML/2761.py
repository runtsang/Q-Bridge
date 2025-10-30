"""Hybrid quantum convolution + classifier, drop‑in replacement for Conv.py.

The class mirrors the classical implementation but uses Qiskit circuits
for both the filter and the classifier.  It exposes the same API so it can
be used interchangeably in pipelines that expect a Conv() factory.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def _build_classifier_circuit(num_qubits: int, depth: int):
    """Return a variational circuit with encoding and ansatz layers."""
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

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

class QuanvCircuit:
    """Quantum convolution filter."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)

        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += self._random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def _random_circuit(self, n, depth):
        """Deterministic placeholder for the original random circuit."""
        circ = QuantumCircuit(n)
        for _ in range(depth):
            for q in range(n):
                circ.rx(np.pi / 4, q)
            for q in range(n - 1):
                circ.cz(q, q + 1)
        return circ

    def run(self, data):
        """Evaluate the filter on classical data."""
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
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class HybridConvClassifier:
    """Quantum hybrid of convolution filter and variational classifier."""
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 127,
        num_qubits: int = 10,
        depth: int = 2,
        backend=None,
        shots: int = 100,
    ) -> None:
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(kernel_size, backend, shots, conv_threshold)
        self.classifier_circuit, self.encoding, self.weights, self.observables = _build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = backend
        self.shots = shots

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the full hybrid pipeline."""
        conv_val = self.conv.run(data)

        # Bind the same feature value to every encoding parameter
        param_binds = {param: conv_val for param in self.encoding}

        job = execute(
            self.classifier_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result().get_counts(self.classifier_circuit)

        # Compute expectation values of the Z observables
        expectations = []
        for obs in self.observables:
            exp = 0.0
            for bitstring, count in result.items():
                # In Qiskit, the rightmost bit corresponds to qubit 0
                qubit_idx = self.observables.index(obs)
                bit = int(bitstring[-(qubit_idx + 1)])
                exp += (1 - 2 * bit) * count
            exp /= self.shots
            expectations.append(exp)

        # Map expectations to a probability vector
        probs = np.array([(e + 1) / 2 for e in expectations])
        return probs

def Conv() -> HybridConvClassifier:
    """Factory that returns a HybridConvClassifier instance."""
    return HybridConvClassifier()

__all__ = ["HybridConvClassifier", "Conv"]
