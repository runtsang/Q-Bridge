"""QuantumClassifierModel: Variational circuit with data re‑uploading and parameter‑shift training."""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import AdamOptimizer
from qiskit.utils import QuantumInstance


class QuantumClassifierModel:
    """A quantum classifier that mirrors the classical interface.

    The circuit is a data‑re‑uploading ansatz with a single variational layer per depth.
    Training uses the parameter‑shift rule and an Adam optimizer on the Aer simulator.
    The class exposes the same metadata (encoding, weight_sizes, observables) as the seed.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        shots: int = 1024,
        backend_name: str = "qasm_simulator",
        optimizer_steps: int = 200,
        learning_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend_name = backend_name
        self.optimizer_steps = optimizer_steps
        self.learning_rate = learning_rate
        self.seed = seed

        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self.backend = Aer.get_backend(self.backend_name)
        self.quantum_instance = QuantumInstance(self.backend, shots=self.shots, seed_simulator=self.seed)
        self.optimizer = AdamOptimizer(steps=self.optimizer_steps, lr=self.learning_rate)

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for i, qubit in enumerate(range(self.num_qubits)):
            qc.rx(encoding[i], qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), [weights], observables

    def _measure(self, qc: QuantumCircuit) -> np.ndarray:
        """Return expectation values for the Z observables of each qubit."""
        job = execute(qc, backend=self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        exp_vals = np.zeros(self.num_qubits)
        for qubit in range(self.num_qubits):
            p0 = 0
            p1 = 0
            for bitstring, c in counts.items():
                # bitstring is MSB first; reverse to match qubit order
                if bitstring[::-1][qubit] == "0":
                    p0 += c
                else:
                    p1 += c
            exp_vals[qubit] = (p0 - p1) / self.shots
        return exp_vals

    def _forward(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute logits for a batch of classical data."""
        logits = np.zeros((X.shape[0], 2))
        for i, x in enumerate(X):
            data_params = {p: val for p, val in zip(self.encoding, x)}
            bound_qc = self.circuit.copy()
            bound_qc = bound_qc.bind_parameters(data_params)
            bound_qc = bound_qc.bind_parameters({self.weights[0][j]: params[j] for j in range(len(params))})
            exp_vals = self._measure(bound_qc)
            logits[i] = exp_vals
        return logits

    def _cross_entropy(self, logits: np.ndarray, y: np.ndarray) -> float:
        log_probs = np.log_softmax(logits, axis=1)
        ce = -np.mean(log_probs[np.arange(len(y)), y])
        return ce

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        params = np.random.randn(self.weights[0].size)
        best_loss = np.inf
        patience = 5
        patience_counter = 0

        for step in range(self.optimizer_steps):
            # Parameter‑shift gradient estimation
            grads = np.zeros_like(params)
            shift = np.pi / 2
            for i in range(len(params)):
                shifted_plus = params.copy()
                shifted_minus = params.copy()
                shifted_plus[i] += shift
                shifted_minus[i] -= shift
                logits_plus = self._forward(shifted_plus, X_train)
                logits_minus = self._forward(shifted_minus, X_train)
                grads[i] = (self._cross_entropy(logits_plus, y_train) -
                            self._cross_entropy(logits_minus, y_train)) / 2

            params = self.optimizer.step(params, grads)

            if X_val is not None and y_val is not None:
                logits_val = self._forward(params, X_val)
                loss = self._cross_entropy(logits_val, y_val)
                if loss < best_loss:
                    best_loss = loss
                    self.params_best = params.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    params = self.params_best
                    break

        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward(self.params, X)
        return np.argmax(logits, axis=1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        logits = self._forward(self.params, X_test)
        loss = self._cross_entropy(logits, y_test)
        acc = np.mean(np.argmax(logits, axis=1) == y_test)
        return loss, acc

    @property
    def metadata(self) -> Tuple[List[ParameterVector], List[int], List[SparsePauliOp]]:
        """Return encoding, weight_sizes, observables for compatibility."""
        return self.encoding, [self.weights[0].size], self.observables


__all__ = ["QuantumClassifierModel"]
