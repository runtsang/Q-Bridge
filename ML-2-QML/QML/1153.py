import numpy as np
from typing import Dict, List, Optional
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, AerPauliExpectation
from qiskit.providers.aer.noise import NoiseModel

class QuantumClassifierModel:
    """
    Parameterized quantum circuit classifier with gradient‑based training
    using the parameter‑shift rule. Supports noise models and early stopping.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 backend: str = "aer_simulator_statevector",
                 shots: int = 1024,
                 seed: Optional[int] = None,
                 noise: Optional[NoiseModel] = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.seed = seed
        self.noise = noise
        self.backend = Aer.get_backend(backend)
        self._build_circuit()
        # Separate parameter vectors for data encoding and variational part
        self.x_params = [p for p in self.circuit.parameters if p.name.startswith("x")]
        self.theta_params = [p for p in self.circuit.parameters if p.name.startswith("theta")]
        # Initialize variational parameters
        self.param_dict: Dict[Parameter, float] = {p: 0.0 for p in self.theta_params}
        self.expectation = AerPauliExpectation()
        # Observable: Z on first qubit
        self.obs = SparsePauliOp("Z" + "I" * (self.num_qubits - 1))

    def _build_circuit(self) -> None:
        """Create a layered ansatz with data encoding and variational layers."""
        self.circuit = QuantumCircuit(self.num_qubits)
        # Data encoding: RX rotations
        x_vec = ParameterVector("x", self.num_qubits)
        for i, q in enumerate(range(self.num_qubits)):
            self.circuit.rx(x_vec[i], q)
        # Variational layers
        theta_vec = ParameterVector("theta", self.num_qubits * self.depth)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                self.circuit.ry(theta_vec[idx], q)
                idx += 1
            for q in range(self.num_qubits - 1):
                self.circuit.cz(q, q + 1)

    def _state_with_params(self, x: np.ndarray, param_dict: Dict[Parameter, float]) -> StateFn:
        """Return the state function for a given input vector and variational parameters."""
        bindings = {p: param_dict[p] for p in self.theta_params}
        bindings.update({p: float(v) for p, v in zip(self.x_params, x)})
        return CircuitStateFn(self.circuit.bind_parameters(bindings))

    def _predict_proba_with_params(self,
                                   X: np.ndarray,
                                   param_dict: Dict[Parameter, float]) -> np.ndarray:
        """Return probability of class 1 for each sample using the provided parameters."""
        probs = []
        for x in X:
            state = self._state_with_params(x, param_dict)
            exp_val = self.expectation.convert(StateFn(self.obs, is_measurement=True) @ state).eval()
            probs.append(float(exp_val))
        return np.array(probs)

    def _loss_with_params(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          param_dict: Dict[Parameter, float]) -> float:
        """Cross‑entropy loss for a batch given explicit parameters."""
        probs = self._predict_proba_with_params(X, param_dict)
        eps = 1e-12
        loss = -np.sum(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)) / len(y)
        return loss

    def _compute_gradients(self,
                           X: np.ndarray,
                           y: np.ndarray) -> Dict[Parameter, float]:
        """Parameter‑shift gradient estimation for all variational parameters."""
        grads: Dict[Parameter, float] = {}
        shift = np.pi / 2
        for param in self.theta_params:
            param_plus = self.param_dict.copy()
            param_minus = self.param_dict.copy()
            param_plus[param] = self.param_dict[param] + shift
            param_minus[param] = self.param_dict[param] - shift
            loss_plus = self._loss_with_params(X, y, param_plus)
            loss_minus = self._loss_with_params(X, y, param_minus)
            grads[param] = (loss_plus - loss_minus) / (2 * shift)
        return grads

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            epochs: int = 20,
            lr: float = 0.01,
            batch_size: int = 32,
            early_stopping_patience: Optional[int] = None) -> Dict[str, List[float]]:
        """Train the circuit using gradient descent and return training loss history."""
        history: Dict[str, List[float]] = {"loss": []}
        best_loss = float("inf")
        patience = early_stopping_patience or 0
        wait = 0

        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            for i in range(0, len(X), batch_size):
                xb = X_shuffled[i:i + batch_size]
                yb = y_shuffled[i:i + batch_size]
                grads = self._compute_gradients(xb, yb)
                for param, grad in grads.items():
                    self.param_dict[param] -= lr * grad
            loss_val = self._loss_with_params(X, y, self.param_dict)
            history["loss"].append(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                wait = 0
            else:
                wait += 1
                if patience and wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 for each sample."""
        return self._predict_proba_with_params(X, self.param_dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels (0/1)."""
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray) -> Dict[str, float]:
        """Compute accuracy and F1 score."""
        preds = self.predict(X)
        acc = (preds == y).mean()
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"accuracy": float(acc), "f1": float(f1)}

__all__ = ["QuantumClassifierModel"]
