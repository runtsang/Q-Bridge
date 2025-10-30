"""
QuantumClassifierModel – Quantum implementation.

This variant builds a variational circuit with separate input and
weight parameters, mirroring the classical depth.  It supports both
classification (Z observables) and regression (Y observables) and
provides a simple prediction interface via Qiskit’s EstimatorQNN.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QuantumClassifierModel:
    """Quantum‑only implementation of the hybrid architecture.

    The class builds a variational circuit with separate input and
    weight parameters, mirrors the classical feed‑forward depth, and
    supports both classification and regression through appropriate
    Pauli observables.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        task: str = "classification",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.task = task

        # Build the circuit
        (
            self.circuit,
            self.input_params,
            self.weight_params,
            self.observables,
        ) = self._build_classifier_circuit(num_qubits, depth, task)

        # Instantiate an estimator
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_classifier_circuit(
        self,
        num_qubits: int,
        depth: int,
        task: str,
    ) -> Tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        if task == "classification":
            observables = [
                SparsePauliOp.from_list(
                    [("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)]
                )
                for i in range(num_qubits)
            ]
        else:  # regression
            observables = [
                SparsePauliOp.from_list(
                    [("I" * i + "Y" + "I" * (num_qubits - i - 1), 1)]
                )
                for i in range(num_qubits)
            ]

        return qc, list(encoding), list(weights), observables

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the circuit on a batch of inputs.

        X must be of shape (n_samples, num_qubits) and contain real
        numbers to be mapped to the input parameters.
        """
        if X.ndim!= 2 or X.shape[1]!= self.num_qubits:
            raise ValueError("Input shape must be (n_samples, num_qubits)")

        predictions = []
        for sample in X:
            param_bindings = {param: val for param, val in zip(self.input_params, sample)}
            # Use the estimator to compute expectation values
            exp_vals = self.estimator_qnn.predict(param_bindings)
            # For regression return the real part of the first observable
            if self.task == "regression":
                predictions.append(exp_vals[0].real)
            else:  # classification
                predictions.append(exp_vals)
        return np.array(predictions)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> None:
        """Gradient‑based training of the weight parameters only.

        This is a lightweight placeholder that uses a simple
        stochastic gradient descent loop.  The loss is defined by the
        task and the circuit parameters are updated via the
        EstimatorQNN’s differentiable interface.
        """
        from qiskit.algorithms.optimizers import COBYLA

        # Prepare a simple objective function
        def loss_fn(theta):
            # Bind weight parameters
            binded = {w: t for w, t in zip(self.weight_params, theta)}
            # Compute expectations for all samples
            preds = []
            for sample in X:
                bindings = {**binded, **{p: v for p, v in zip(self.input_params, sample)}}
                exp_vals = self.estimator_qnn.predict(bindings)
                preds.append(exp_vals[0].real if self.task == "regression" else exp_vals)
            preds = np.array(preds)
            if self.task == "classification":
                probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
                return -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))
            else:
                return np.mean((preds - y) ** 2)

        opt = COBYLA(maxiter=epochs)
        opt.optimize(num_vars=len(self.weight_params), objective_function=loss_fn)

__all__ = ["QuantumClassifierModel"]
