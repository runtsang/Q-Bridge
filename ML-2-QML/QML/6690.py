import numpy as np
from typing import Tuple, Iterable, List
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator

class QuantumClassifier:
    """
    Quantum classifier that extends the original interface with
    a parameter‑shift training routine and hybrid evaluation.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 backend: str = 'aer_simulator',
                 shots: int = 1024,
                 seed: int = 42):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.seed = seed

        self.circuit, self.encoding_params, self.weights_params, self.observables = self.build_classifier_circuit(num_qubits, depth)
        self.backend = self._get_backend(backend)
        self.parameter_shift_step = 0.01

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Create a layered ansatz with data encoding and variational layers.
        Returns the circuit, encoding parameters, weight parameters and observables.
        """
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

    def _get_backend(self, name: str) -> AerSimulator:
        """
        Return a Qiskit Aer simulator instance based on the requested backend name.
        """
        if name == 'qasm_simulator':
            return Aer.get_backend('qasm_simulator')
        if name =='statevector_simulator':
            return Aer.get_backend('statevector_simulator')
        return Aer.get_backend('aer_simulator')

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of data points.
        `data` should be of shape (batch, num_qubits) with values in [-π, π].
        Returns expectation values of the observables.
        """
        batch_size = data.shape[0]
        expectations = np.zeros((batch_size, self.num_qubits))
        for i, sample in enumerate(data):
            bound_circuit = self.circuit.bind_parameters(
                {p: val for p, val in zip(self.encoding_params, sample)}
            )
            result = execute(bound_circuit, self.backend, shots=self.shots).result()
            counts = result.get_counts()
            for j, obs in enumerate(self.observables):
                exp = self._expectation_from_counts(counts, obs)
                expectations[i, j] = exp
        return expectations

    def _expectation_from_counts(self, counts: dict, pauli: SparsePauliOp) -> float:
        """
        Compute expectation value of a Pauli operator from measurement counts.
        """
        exp = 0.0
        for bitstring, cnt in counts.items():
            eigenvalue = 1
            for idx, pauli_char in enumerate(pauli.to_label()):
                if pauli_char == "Z":
                    eigenvalue *= 1 if bitstring[self.num_qubits - idx - 1] == "0" else -1
            exp += eigenvalue * cnt
        return exp / sum(counts.values())

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, lr: float = 0.1) -> None:
        """
        Simple gradient descent using the parameter‑shift rule.
        Assumes binary classification with labels 0/1.
        """
        for epoch in range(epochs):
            grads = np.zeros(len(self.weights_params), dtype=float)
            for idx, param in enumerate(self.weights_params):
                plus = self._shifted_expectation(X, idx, +self.parameter_shift_step)
                minus = self._shifted_expectation(X, idx, -self.parameter_shift_step)
                grads[idx] = (plus - minus) / (2 * self.parameter_shift_step)

            # Update weights
            for i, param in enumerate(self.weights_params):
                new_val = float(param) + lr * grads[i]
                self.weights_params[i] = new_val

    def _shifted_expectation(self, X: np.ndarray, idx: int, shift: float) -> float:
        """
        Compute average expectation value over the dataset with a shifted parameter.
        """
        expectations = []
        for sample in X:
            # Bind encoding parameters
            bound_circuit = self.circuit.bind_parameters(
                {p: val for p, val in zip(self.encoding_params, sample)}
            )
            # Shift the targeted weight
            shifted_params = {p: float(p) for p in self.weights_params}
            shifted_params[self.weights_params[idx]] += shift
            for p, val in shifted_params.items():
                bound_circuit.set_parameter(p, val)
            result = execute(bound_circuit, self.backend, shots=self.shots).result()
            counts = result.get_counts()
            exp = self._expectation_from_counts(counts, self.observables[0])
            expectations.append(exp)
        return np.mean(expectations)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by thresholding the first observable expectation.
        """
        expectations = self.execute(X)
        probs = 1 / (1 + np.exp(-expectations[:, 0]))
        return (probs > 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy on the given dataset.
        """
        preds = self.predict(X)
        return np.mean(preds == y)

__all__ = ["QuantumClassifier"]
