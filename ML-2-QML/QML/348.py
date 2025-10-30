"""QuantumClassifierModel: Quantum implementation with variational circuit and gradient‑based training."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional
import numpy as np
import math
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """Variational quantum classifier with a simple data‑uploading ansatz.

    The class follows the same public API as the classical counterpart.
    It supports training on a simulator backend using the parameter‑shift
    rule and can be extended to real hardware by swapping the Aer
    backend.  The interface deliberately mirrors the PyTorch version
    so that experiments can be run in a hybrid setting.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 backend: Optional[object] = None,
                 shots: int = 1024,
                 optimizer: str = "COBYLA",
                 seed: int = 42) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.optimizer_name = optimizer
        self.rng = np.random.default_rng(seed)

        # Circuit template
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self.params = self.rng.normal(size=len(self.weights))

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Construct a layered ansatz with data encoding."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables (Z on each qubit)
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def _parameter_shift(self, params: np.ndarray, func, i: int) -> float:
        """Compute gradient via parameter‑shift rule."""
        shift = math.pi / 2
        plus = params.copy()
        minus = params.copy()
        plus[i] += shift
        minus[i] -= shift
        return (func(plus) - func(minus)) / (2 * shift)

    def _cost(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Cross‑entropy cost using expectation values as logits."""
        logits = self._predict_with_params(params, X)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        ce = -np.mean(np.sum(np.eye(2)[y] * np.log(probs + 1e-12), axis=1))
        return ce

    def _predict_with_params(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return expectation values for each sample."""
        n_samples = X.shape[0]
        logits = np.zeros((n_samples, self.num_qubits))
        for i, sample in enumerate(X):
            bound_qc = self.circuit.bind_parameters(
                dict(zip(self.encoding, sample))
            ).bind_parameters(
                dict(zip(self.weights, params))
            )
            job = execute(bound_qc, self.backend, shots=self.shots, memory=True)
            result = job.result()
            counts = result.get_counts()
            for qubit in range(self.num_qubits):
                exp = 0.0
                for bitstring, cnt in counts.items():
                    bit = int(bitstring[::-1][qubit])
                    exp += (1 - 2 * bit) * cnt / self.shots
                logits[i, qubit] = exp
        return logits

    def fit(self, X: np.ndarray, y: np.ndarray,
            max_iter: int = 200,
            tolerance: float = 1e-3,
            verbose: bool = False) -> None:
        """Train using COBYLA (or SPSA) with parameter‑shift gradients."""
        from scipy.optimize import minimize

        def objective(params):
            return self._cost(params, X, y)

        init_params = self.rng.normal(size=len(self.params))
        res = minimize(objective,
                       init_params,
                       method=self.optimizer_name,
                       options={"maxiter": max_iter, "disp": verbose})

        self.params = res.x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class indices based on sign of expectation values."""
        logits = self._predict_with_params(self.params, X)
        preds = (logits.sum(axis=1) >= 0).astype(int)
        return preds

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Return accuracy and cross‑entropy loss."""
        preds = self.predict(X)
        acc = (preds == y).mean()
        loss = self._cost(self.params, X, y)
        return acc, loss

    def get_metadata(self) -> Tuple[List[int], List[int], List[SparsePauliOp]]:
        """Return encoding indices, weight sizes, and observables."""
        return list(self.encoding), [len(p) for p in self.params], self.observables
