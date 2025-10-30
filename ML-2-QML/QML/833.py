"""Quantum classifier using Qiskit with a parameter‑shift gradient and hybrid training."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from sklearn.model_selection import KFold

class QuantumClassifier:
    """
    Quantum classifier mirroring the classical interface.
    Uses a simple data‑uploading ansatz, a parameter‑shift gradient,
    and a hybrid classical optimizer (Adam).  The API is
    intentionally compatible with the classical counterpart.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 3,
                 lr: float = 0.1,
                 epochs: int = 200,
                 batch_size: int = 32,
                 seed: int | None = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self.params = np.random.uniform(-np.pi, np.pi, size=len(self.encoding) + len(self.weights))
        self.optimizer = self._adam_optimizer()
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
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
        return qc, encoding, weights, observables

    def _adam_optimizer(self):
        return {"m": np.zeros(len(self.params)), "v": np.zeros(len(self.params)), "t": 0}

    def _adam_step(self, grads: np.ndarray):
        opt = self.optimizer
        opt["t"] += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr_t = self.lr * np.sqrt(1 - beta2 ** opt["t"]) / (1 - beta1 ** opt["t"])
        opt["m"] = beta1 * opt["m"] + (1 - beta1) * grads
        opt["v"] = beta2 * opt["v"] + (1 - beta2) * (grads ** 2)
        m_hat = opt["m"] / (1 - beta1 ** opt["t"])
        v_hat = opt["v"] / (1 - beta2 ** opt["t"])
        update = lr_t * m_hat / (np.sqrt(v_hat) + eps)
        return update

    def _expectation_single(self, params: np.ndarray, sample: np.ndarray) -> float:
        bound_qc = self.circuit.copy()
        # bind encoding parameters
        bound_qc.assign_parameters({self.encoding[i]: val for i, val in enumerate(sample)}, inplace=True)
        # bind variational parameters
        bound_qc.bind_parameters(dict(zip(self.weights, params[len(self.encoding):])), inplace=True)
        compiled = transpile(bound_qc, self.backend)
        job = self.backend.run(compiled, shots=1024)
        result = job.result()
        counts = result.get_counts()
        probs = {state: count / 1024 for state, count in counts.items()}
        # expectation of first qubit Z
        exp_val = 0.0
        for state, p in probs.items():
            bit = int(state[-1])  # qubit 0 is the least significant bit
            eig = 1 if bit == 0 else -1
            exp_val += eig * p
        return exp_val

    def _parameter_shift_gradient_single(self, params: np.ndarray, sample: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(params)
        shift = np.pi / 2
        for i in range(len(params)):
            shift_vec = np.zeros_like(params)
            shift_vec[i] = shift
            f_plus = self._expectation_single(params + shift_vec, sample)
            f_minus = self._expectation_single(params - shift_vec, sample)
            grad[i] = (f_plus - f_minus) / 2
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray, *,
            val_split: float = 0.1, shuffle: bool = True) -> None:
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        n_val = int(n_samples * val_split)
        train_idx, val_idx = indices[:n_val], indices[n_val:]

        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # training loop
            for start in range(0, n_val, self.batch_size):
                end = min(start + self.batch_size, n_val)
                batch_idx = train_idx[start:end]
                grads = np.zeros_like(self.params)
                for i, idx in enumerate(batch_idx):
                    sample = X[idx]
                    label = y[idx]
                    exp_val = self._expectation_single(self.params, sample)
                    p = (exp_val + 1) / 2
                    if label == 1:
                        dL_dexp = -1 / (p + 1e-12) * 0.5
                    else:
                        dL_dexp = 1 / (1 - p + 1e-12) * 0.5
                    grad_exp = self._parameter_shift_gradient_single(self.params, sample)
                    grads += dL_dexp * grad_exp
                grads /= len(batch_idx)
                updates = self._adam_step(grads)
                self.params -= updates

            # validation
            val_loss = self._validation_loss(self.params, X[val_idx], y[val_idx])
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = self.params.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.params = best_params

    def _validation_loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        loss = 0.0
        for sample, label in zip(X, y):
            exp_val = self._expectation_single(params, sample)
            p = (exp_val + 1) / 2
            if label == 1:
                loss -= np.log(p + 1e-12)
            else:
                loss -= np.log(1 - p + 1e-12)
        return loss / len(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for sample in X:
            exp_val = self._expectation_single(self.params, sample)
            p = (exp_val + 1) / 2
            probs.append([1 - p, p])
        return np.array(probs)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict(X)
        preds = np.argmax(probs, axis=1)
        return np.mean(preds == y)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> List[float]:
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        scores = []
        for train_idx, val_idx in kf.split(X):
            self.fit(X[train_idx], y[train_idx], val_split=0.0)
            scores.append(self.score(X[val_idx], y[val_idx]))
        return scores

__all__ = ["QuantumClassifier"]
