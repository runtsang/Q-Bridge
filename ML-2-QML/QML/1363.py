"""Quantum classifier circuit with a richer ansatz and multi‑qubit observables.

The class follows the same public contract as its classical counterpart:
`metadata()` returns the encoding, parameter vector, and observable list used
for training and inference.  The circuit now contains an additional entangling
block (RZZ + CNOT) per depth layer, yielding a more expressive parameterised
state that can capture higher‑order correlations.

The implementation uses Qiskit for circuit construction and the state‑vector
simulator for expectation evaluation.  The helper `parameter_shift` routine
computes gradients analytically, enabling a gradient‑based optimiser.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifier:
    """
    Quantum circuit factory mirroring the classical `QuantumClassifier` API.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) in the circuit.
    depth : int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth * 2)  # RY and RZZ
        self.circuit, self.obs = self._build_circuit()

        # expose metadata
        self.encoding_list: List[ParameterVector] = list(self.encoding)
        self.weights_list: List[ParameterVector] = list(self.weights)
        self.observables: List[SparsePauliOp] = self.obs

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[SparsePauliOp]]:
        """Construct the variational ansatz."""
        qc = QuantumCircuit(self.num_qubits)

        # Feature encoding
        for i, param in enumerate(self.encoding):
            qc.rx(param, i)

        weight_idx = 0
        for _ in range(self.depth):
            # Single‑qubit rotations
            for i in range(self.num_qubits):
                qc.ry(self.weights[weight_idx], i)
                weight_idx += 1
            # Entangling block: RZZ + CNOT chain
            for i in range(self.num_qubits - 1):
                qc.rzz(np.pi / 4, i, i + 1)
                qc.cx(i, i + 1)
            # Optional reverse entanglement for symmetry
            for i in reversed(range(self.num_qubits - 1)):
                qc.cx(i, i + 1)
                qc.rzz(np.pi / 4, i, i + 1)

        # Measurement observables: single‑qubit Z and two‑qubit ZZ terms
        observables = [
            SparsePauliOp("Z" * i + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        # add neighbouring ZZ terms
        for i in range(self.num_qubits - 1):
            pauli_str = "I" * i + "ZZ" + "I" * (self.num_qubits - i - 2)
            observables.append(SparsePauliOp(pauli_str))

        return qc, observables

    def metadata(self) -> Tuple[Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Return (encoding, weights, observables) for compatibility."""
        return self.encoding_list, self.weights_list, self.observables

    # ------------------------------------------------------------------
    # Simple simulation helpers
    # ------------------------------------------------------------------
    def _statevector(self, x: np.ndarray) -> np.ndarray:
        """Return the state‑vector for a given feature vector."""
        param_binds = {str(p): val for p, val in zip(self.encoding, x)}
        bound_qc = self.circuit.bind_parameters(param_binds)
        backend = AerSimulator(method="statevector")
        result = execute(bound_qc, backend).result()
        return result.get_statevector(bound_qc)

    def expectation(self, x: np.ndarray, observable: SparsePauliOp) -> float:
        """Compute the expectation value of a single observable."""
        sv = self._statevector(x)
        return float(np.vdot(sv, observable.to_matrix().dot(sv)).real)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for a batch of feature vectors."""
        preds = []
        for x in X:
            exp_vals = [self.expectation(x, obs) for obs in self.observables]
            # simple linear readout: weight vector [1, -1] over observables
            logits = np.array([sum(exp_vals), -sum(exp_vals)])
            preds.append(np.argmax(logits))
        return np.array(preds)

    # ------------------------------------------------------------------
    # Gradient‑based optimisation using the parameter‑shift rule
    # ------------------------------------------------------------------
    def parameter_shift_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[np.ndarray]:
        """Return gradients of a simple cross‑entropy loss w.r.t. all variational parameters."""
        eps = np.pi / 2
        grads = []
        for idx, param in enumerate(self.weights):
            shift_plus = np.array(self.weights, copy=True)
            shift_minus = np.array(self.weights, copy=True)
            shift_plus[idx] += eps
            shift_minus[idx] -= eps

            loss_plus = self._loss(X, y, shift_plus)
            loss_minus = self._loss(X, y, shift_minus)
            grads.append((loss_plus - loss_minus) / (2 * np.sin(eps)))
        return grads

    def _loss(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Cross‑entropy loss for a batch given a specific weight vector."""
        logits = []
        for x in X:
            exp_vals = [self.expectation(x, obs) for obs in self.observables]
            logits.append(np.array([sum(exp_vals), -sum(exp_vals)]))
        logits = np.stack(logits)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[range(len(y)), y]))
        return loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Gradient‑based training using the parameter‑shift rule."""
        for epoch in range(epochs):
            grads = self.parameter_shift_gradients(X, y)
            self.weights = np.array(self.weights) - lr * np.array(grads)
            if verbose and (epoch + 1) % 5 == 0:
                loss = self._loss(X, y, self.weights)
                acc = (self.predict(X) == y).mean()
                print(f"Epoch {epoch+1:03d} | loss={loss:.4f} | acc={acc:.4f}")

__all__ = ["QuantumClassifier"]
