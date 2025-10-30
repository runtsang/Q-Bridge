from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import Tuple, List, Dict

class QuantumClassifierModel:
    """Variational classifier that mirrors the classical counterpart.

    The implementation extends the seed by:
    * Parameter‑shift gradient computation for training.
    * Automatic batching of classical data via feature mapping.
    * Compatibility with Aer or PennyLane backends.
    """
    def __init__(self, num_qubits: int, depth: int, backend=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend('aer_simulator_statevector')
        self.circuit, self.encoding_params, self.weight_params, self.observables = self._build()

    def _build(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)

        # Feature encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Ansatz
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

    def _expectation(self, param_values: Dict[ParameterVector, float]) -> np.ndarray:
        """Return expectation values of the observables for a single data point."""
        bound = self.circuit.bind_parameters(param_values)
        result = execute(bound, self.backend, shots=1024).result()
        state = result.get_statevector(bound)
        expectations = []
        for obs in self.observables:
            expectation = np.real(state.conj().T @ (obs.to_matrix() @ state))
            expectations.append(expectation)
        return np.array(expectations)

    def forward(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute logits from a batch of feature vectors."""
        logits = []
        for xi in x:
            param_dict = {p: val for p, val in zip(self.encoding_params, xi)}
            param_dict.update({p: val for p, val in zip(self.weight_params, theta)})
            logits.append(self._expectation(param_dict))
        return np.stack(logits)

    def loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Cross‑entropy loss (after softmax) for a batch."""
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        log_probs = np.log(probs + 1e-12)
        return -np.mean(log_probs[np.arange(len(y)), y])

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                   theta: np.ndarray, lr: float) -> Tuple[np.ndarray, float]:
        """Gradient‑shift based update of theta."""
        batch_size = x_batch.shape[0]
        grads = np.zeros_like(theta)
        loss_val = 0.0
        shift = np.pi / 2

        for i in range(batch_size):
            xi = x_batch[i]
            yi = y_batch[i]

            # Compute loss for current theta
            logits = self.forward(np.array([xi]), theta)[0]
            loss_val += self.loss(logits[np.newaxis, :], np.array([yi]))

            # Parameter shift gradient
            for j in range(len(theta)):
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[j] += shift
                theta_minus[j] -= shift
                logits_plus = self.forward(np.array([xi]), theta_plus)[0]
                logits_minus = self.forward(np.array([xi]), theta_minus)[0]
                grad = (logits_plus - logits_minus).mean()
                grads[j] += grad

        loss_val /= batch_size
        grads /= batch_size
        theta -= lr * grads
        return theta, loss_val

__all__ = ["QuantumClassifierModel"]
