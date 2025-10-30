import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, List, Tuple

class QuantumClassifierModel:
    """
    Variational quantum classifier with a data‑encoding layer and trainable ansatz.
    Provides fit, predict, and metadata methods compatible with the classical API.
    """
    def __init__(self, num_qubits: int, depth: int):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding_params, self.variational_params, self.observables = self._build_circuit()
        self.param_vector = np.concatenate([self.encoding_params, self.variational_params])
        self.backend = Aer.get_backend("statevector_simulator")

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)

        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def _evaluate(self, params: np.ndarray) -> np.ndarray:
        """
        Compute expectation values of the observables for a single data point.
        """
        bind_map = {self.encoding_params[i]: params[i] for i in range(self.num_qubits)}
        for i, p in enumerate(self.variational_params):
            bind_map[self.variational_params[i]] = params[self.num_qubits + i]
        bound_circuit = self.circuit.bind_parameters(bind_map)
        job = execute(bound_circuit, self.backend)
        state = job.result().get_statevector(bound_circuit)
        return np.array([op.expectation_value(state).real for op in self.observables])

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X, params=params)
        return np.mean(preds!= y)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 0.01) -> None:
        """
        Train the variational parameters using a crude finite‑difference gradient.
        """
        for _ in range(epochs):
            grads = np.zeros(len(self.param_vector))
            for i in range(len(self.param_vector)):
                eps = 1e-6
                plus = self.param_vector.copy()
                minus = self.param_vector.copy()
                plus[i] += eps
                minus[i] -= eps
                grads[i] = (self._loss(plus, X, y) - self._loss(minus, X, y)) / (2 * eps)
            self.param_vector -= lr * grads

    def predict(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """
        Predict labels for a batch of data points.
        """
        if params is None:
            params = self.param_vector
        preds = []
        for x in X:
            exp_vals = self._evaluate(np.concatenate([x, params[self.num_qubits:]]))
            preds.append(np.argmax(exp_vals))
        return np.array(preds)

    def get_metadata(self) -> Tuple[List[int], List[int], List[SparsePauliOp]]:
        """
        Return encoding indices, parameter sizes, and observables.
        """
        enc_sizes = [self.num_qubits]
        var_sizes = [self.num_qubits * self.depth]
        return list(range(self.num_qubits)), enc_sizes + var_sizes, self.observables

__all__ = ["QuantumClassifierModel"]
