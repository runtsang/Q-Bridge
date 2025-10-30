"""Quantum variational classifier that mirrors the classical helper.

The class builds a data‑encoding layer followed by a depth‑controlled
variational ansatz.  It exposes a simple expectation evaluation
and a lightweight L‑BFGS training routine.  A static factory
`build_classifier_circuit` returns the same tuple as the seed,
allowing seamless integration with classical pipelines.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


class QuantumClassifierModel:
    """
    Variational circuit classifier.

    Parameters
    ----------
    num_qubits : int
        The number of qubits (analogous to input features).
    depth : int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

        # Build the circuit once; parameters are stored as ParameterVectors
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return circuit, encoding vector, parameter vector, and observables."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data‑encoding layer
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

        # Observables are Pauli‑Z on each qubit
        obs = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, list(encoding), list(weights), obs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_circuit(self, params: np.ndarray | None = None) -> QuantumCircuit:
        """Return a parameterised circuit.  If `params` is supplied, bind them."""
        qc = self.circuit.copy()
        if params is not None:
            bind_dict = {str(p): float(v) for p, v in zip(self.weights, params)}
            qc = qc.bind_parameters(bind_dict)
        return qc

    def get_encoding(self) -> List[int]:
        """Return encoding indices (here simply 0..num_qubits‑1)."""
        return list(range(self.num_qubits))

    def get_weights(self) -> List[ParameterVector]:
        """Return the parameter vector for the variational layers."""
        return self.weights

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the measurement observables."""
        return self.observables

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def expectation(self, params: np.ndarray, shots: int = 1) -> np.ndarray:
        """Return expectation values for each observable given `params`."""
        qc = self.get_circuit(params)
        backend = Aer.get_backend("aer_simulator")
        job = execute(
            qc,
            backend=backend,
            shots=shots,
            parameter_binds=[{str(p): float(v) for p, v in zip(self.weights, params)}],
        )
        state = job.result().get_statevector(qc)
        exp = np.array(
            [np.real(state.conj().T @ op.to_matrix() @ state) for op in self.observables]
        )
        return exp

    # ------------------------------------------------------------------
    # Simple training loop using scipy's L-BFGS-B
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 20,
        bound: Tuple[float, float] = (-np.pi, np.pi),
    ) -> np.ndarray:
        """
        Train the variational parameters to minimise a binary cross‑entropy
        loss between the expectation of the first observable and the labels.
        """
        num_params = len(self.weights)
        init = np.zeros(num_params)

        def loss_fn(theta):
            exp_vals = self.expectation(theta)
            probs = 1 / (1 + np.exp(-exp_vals[0]))  # sigmoid of first observable
            eps = 1e-12
            loss = -np.mean(
                y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)
            )
            return loss

        opt = minimize(
            loss_fn, init, method="L-BFGS-B", bounds=[bound] * num_params, options={"maxiter": epochs}
        )
        self.params = opt.x
        return self.params

    # ------------------------------------------------------------------
    # Convenience wrapper matching the seed function signature
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Static factory that returns the same tuple as the original seed
        function but with a richer variational ansatz.
        """
        model = QuantumClassifierModel(num_qubits, depth)
        return model.circuit, list(model.encoding), list(model.weights), model.observables
