import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical interface.
    Uses a data‑re‑uploading ansatz and parameter‑shift gradients.
    """

    def __init__(self, num_qubits: int, depth: int = 2, backend=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend('aer_simulator_statevector')
        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit(num_qubits, depth)

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
        """
        Create a data‑re‑uploading ansatz with RX encoding and a single‑qubit Z observable.
        """
        enc = ParameterVector("x", num_qubits)
        theta = ParameterVector("theta", num_qubits * depth)
        qc = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            qc.rx(enc[q], q)

        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qc.ry(theta[idx], q)
                idx += 1
            for q in range(num_qubits - 1):
                qc.cz(q, q + 1)

        obs = [SparsePauliOp("Z" + "I" * (num_qubits - 1))]
        return qc, list(enc), list(theta), obs

    def _expectation(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Return the expectation value of the first‑qubit Z operator for each sample.
        """
        n_samples = len(X)
        expectations = np.zeros(n_samples)
        for i, x in enumerate(X):
            bound = self.circuit.bind_parameters({p: val for p, val in zip(self.encoding, x)})
            for w, val in zip(self.weights, theta):
                bound = bound.bind_parameters({w: val})
            job = execute(bound, self.backend, shots=1024)
            result = job.result()
            state = np.asarray(result.get_statevector(bound))
            exp = np.real(np.vdot(state, self.observables[0].to_matrix() @ state))
            expectations[i] = exp
        return expectations

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 20, verbose: bool = True) -> None:
        """
        Optimize parameters with a simple gradient descent using the parameter‑shift rule.
        """
        theta = np.random.randn(len(self.weights))
        for epoch in range(1, epochs + 1):
            loss = 0.0
            grads = np.zeros_like(theta)
            expectations = self._expectation(X, theta)
            probs = (1 - expectations) / 2
            for i, yi in enumerate(y):
                loss += -np.log(probs[i] if yi == 1 else 1 - probs[i])
                # parameter‑shift gradient
                for p_idx in range(len(theta)):
                    shift = 0.5
                    theta_pos = theta.copy()
                    theta_pos[p_idx] += shift
                    theta_neg = theta.copy()
                    theta_neg[p_idx] -= shift
                    exp_pos = self._expectation(X, theta_pos)
                    exp_neg = self._expectation(X, theta_neg)
                    grad = (exp_pos - exp_neg).mean() / (2 * shift)
                    grads[p_idx] += grad
            grads /= len(X)
            theta -= lr * grads
            loss /= len(X)
            if verbose:
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")
        self.weights = theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for each sample.
        """
        expectations = self._expectation(X, self.weights)
        probs = (1 - expectations) / 2
        return np.vstack([1 - probs, probs]).T
