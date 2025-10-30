"""Quantum classifier with parameter‑shift training and flexible ansatz."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifierModelGen:
    """Variational quantum classifier that mirrors the classical interface.

    Parameters
    ----------
    num_qubits
        Number of qubits / input feature dimension.
    depth
        Number of variational layers.
    encoding
        Single‑qubit gate used for data encoding (``rx``, ``ry`` or ``rz``).
    entanglement
        Entangling pattern: ``cnot`` (nearest‑neighbour), ``cz`` or ``all``.
    shots
        Number of shots used for measurement statistics.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        encoding: str = "rx",
        entanglement: str = "cnot",
        shots: int = 1024,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = encoding
        self.entanglement = entanglement
        self.shots = shots
        self.backend = AerSimulator()
        self.params = np.zeros(num_qubits * depth)

        # Build the ansatz
        self.circuit, self.enc_params, self.var_params, self.observables = self.build_classifier_circuit(
            num_qubits, depth, encoding=encoding, entanglement=entanglement
        )

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        encoding: str = "rx",
        entanglement: str = "cnot",
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return a variational circuit and its metadata.

        The circuit consists of an encoding layer followed by ``depth`` layers of
        single‑qubit rotations and a configurable entangling pattern.
        Observables are single‑qubit Z operators on each qubit.
        """
        enc_params = ParameterVector("x", num_qubits)
        var_params = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)

        # Data encoding
        for i in range(num_qubits):
            if encoding == "rx":
                qc.rx(enc_params[i], i)
            elif encoding == "ry":
                qc.ry(enc_params[i], i)
            elif encoding == "rz":
                qc.rz(enc_params[i], i)
            else:
                qc.rx(enc_params[i], i)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qc.ry(var_params[idx], q)
                idx += 1
            if entanglement == "cnot":
                for q in range(num_qubits - 1):
                    qc.cx(q, q + 1)
            elif entanglement == "cz":
                for q in range(num_qubits - 1):
                    qc.cz(q, q + 1)
            elif entanglement == "all":
                for q in range(num_qubits):
                    for r in range(q + 1, num_qubits):
                        qc.cx(q, r)

        # Observables
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return qc, list(enc_params), list(var_params), observables

    # --------------------------------------------------------------------------- #
    # Core quantum evaluation helpers
    # --------------------------------------------------------------------------- #

    def _expectation(self, circuit: QuantumCircuit, params: np.ndarray) -> np.ndarray:
        """Return expectation values of the observables for a given parameter vector."""
        # Bind variational parameters
        bound = circuit.bind_parameters(
            {p: float(v) for p, v in zip(self.var_params, params)}
        )
        # Add measurement
        bound_measure = bound.copy()
        bound_measure.measure_all()
        # Run simulation
        transpiled = transpile(bound_measure, self.backend)
        job = self.backend.run(transpiled, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        exp_vals = np.zeros(len(self.observables))
        for i, _ in enumerate(self.observables):
            # expectation of Z = (#0 - #1) / shots
            zero = sum(val for key, val in counts.items() if key[::-1][i] == "0")
            one = self.shots - zero
            exp_vals[i] = (zero - one) / self.shots
        return exp_vals

    def _parameter_shift_grad(self, circuit: QuantumCircuit, params: np.ndarray) -> np.ndarray:
        """Compute the Jacobian of the expectation vector with respect to the parameters."""
        shift = np.pi / 2
        num_params = len(params)
        num_obs = len(self.observables)
        grad = np.zeros((num_params, num_obs))
        for k in range(num_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[k] += shift
            params_minus[k] -= shift
            f_plus = self._expectation(circuit, params_plus)
            f_minus = self._expectation(circuit, params_minus)
            grad[k] = (f_plus - f_minus) / 2
        return grad

    # --------------------------------------------------------------------------- #
    # Training / inference
    # --------------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        lr: float = 0.1,
    ) -> None:
        """Gradient‑descent training using the parameter‑shift rule."""
        params = np.random.randn(len(self.var_params))
        for _ in range(epochs):
            for x, label in zip(X, y):
                # Bind data
                bound = self.circuit.copy()
                bound = bound.bind_parameters(
                    {p: float(v) for p, v in zip(self.enc_params, x)}
                )
                # Forward
                exp_vals = self._expectation(bound, params)
                logits = exp_vals
                probs = np.exp(logits) / np.sum(np.exp(logits))
                loss = -np.log(probs[label] + 1e-8)
                # Gradients
                grad_exp = self._parameter_shift_grad(bound, params)  # shape (n_params, n_obs)
                grad_logits = probs
                grad_logits[label] -= 1
                grad_loss = grad_exp.T @ grad_logits  # shape (n_params,)
                params -= lr * grad_loss
        self.params = params
        # Bind optimized parameters to the circuit for inference
        self.circuit = self.circuit.bind_parameters(
            {p: float(v) for p, v in zip(self.var_params, self.params)}
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for ``X``."""
        preds = []
        for x in X:
            bound = self.circuit.copy()
            bound = bound.bind_parameters(
                {p: float(v) for p, v in zip(self.enc_params, x)}
            )
            exp_vals = self._expectation(bound, self.params)
            preds.append(int(np.argmax(exp_vals)))
        return np.array(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return average cross‑entropy loss on a held‑out set."""
        losses = []
        for x, label in zip(X, y):
            bound = self.circuit.copy()
            bound = bound.bind_parameters(
                {p: float(v) for p, v in zip(self.enc_params, x)}
            )
            exp_vals = self._expectation(bound, self.params)
            logits = exp_vals
            probs = np.exp(logits) / np.sum(np.exp(logits))
            loss = -np.log(probs[label] + 1e-8)
            losses.append(loss)
        return np.mean(losses)


__all__ = ["QuantumClassifierModelGen"]
