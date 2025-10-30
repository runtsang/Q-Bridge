import numpy as np
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA

class QuantumClassifierModel:
    """Quantum classifier with a tunable feature‑map depth and a parameter‑shift optimiser."""
    def __init__(
        self,
        num_qubits: int,
        depth: int = 1,
        learning_rate: float = 0.01,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Parameter vectors
        self.encoding_params = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("theta", num_qubits * depth)

        # Build base circuit
        self.circuit = self._build_circuit()

        # Observables: Z on each qubit
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        # Simulator
        self.sim = AerSimulator(method="statevector")

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Feature map
        for qubit, param in enumerate(self.encoding_params):
            qc.rx(param, qubit)
        # Ansatz
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weight_params[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    def _expectation(self, X_batch: np.ndarray) -> np.ndarray:
        """Return expectation values for each observable over the batch."""
        n = X_batch.shape[0]
        expectations = np.zeros((n, self.num_qubits))
        for i, sample in enumerate(X_batch):
            param_dict = {p: sample[j] for j, p in enumerate(self.encoding_params)}
            for j, w in enumerate(self.weight_params):
                param_dict[w] = self.weights[j]
            bound = self.circuit.bind_parameters(param_dict)
            state = Statevector.from_instruction(bound)
            for k, obs in enumerate(self.observables):
                expectations[i, k] = state.expectation_value(obs).real
        return expectations

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Compute logits as linear combination of expectation values."""
        expect = self._expectation(X)
        logits = np.zeros((X.shape[0], 2))
        if self.num_qubits >= 2:
            logits[:, 0] = expect[:, 0]
            logits[:, 1] = expect[:, 1]
        else:
            logits[:, 0] = expect[:, 0]
            logits[:, 1] = 0.0
        return logits

    def _objective(self, weights_flat: np.ndarray) -> float:
        """Cross‑entropy loss over the training data."""
        self.weights = weights_flat
        X_train, y_train = self.X_train, self.y_train
        logits = self._predict_logits(X_train)
        # Softmax cross‑entropy
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(len(y_train)), y_train] + 1e-12))
        return ce

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumClassifierModel":
        """Train using SPSA with a parameter‑shift gradient estimator."""
        self.X_train = X
        self.y_train = y
        # Initialise weights randomly
        self.weights = np.random.randn(self.num_qubits * self.depth)

        opt = SPSA(maxiter=self.epochs, learning_rate=self.learning_rate)
        opt.minimize(
            fun=self._objective,
            initial_point=self.weights,
            callback=None,
        )
        self.weights = opt.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

def build_classifier_circuit(
    num_qubits: int,
    depth: int = 1,
    learning_rate: float = 0.01,
    epochs: int = 50,
    batch_size: int = 32,
) -> Tuple[QuantumClassifierModel, List[int], List[int], List[SparsePauliOp]]:
    model = QuantumClassifierModel(num_qubits, depth, learning_rate, epochs, batch_size)
    encoding = list(range(num_qubits))
    weight_sizes = [1] * (num_qubits * depth)  # placeholder
    observables = model.observables
    return model, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
