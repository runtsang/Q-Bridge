"""Quantum‑centric hybrid classifier.

This module implements :class:`HybridClassifier` using Qiskit’s
``EstimatorQNN`` to embed a variational circuit into a neural‑network
interface.  The design reuses the data‑uploading ansatz from the seed
``QuantumClassifierModel`` and the lightweight regression network from
``EstimatorQNN``.  The estimator is instantiated with a
``StatevectorEstimator`` backend, making the model fully differentiable
and ready for variational training on a quantum simulator or a real device.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


class HybridClassifier:
    """
    Hybrid quantum‑classical classifier built on Qiskit’s EstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the variational circuit.
    depth : int
        Depth of the ansatz.
    num_features : int
        Dimensionality of the classical feature vector (must match
        ``num_qubits`` for data‑uploading encoding).
    """

    def __init__(self, num_qubits: int, depth: int, num_features: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_features = num_features

        # Build the quantum circuit
        (
            self.quantum_circuit,
            self.encoding,
            self.weight_params,
            self.observables,
        ) = self._build_quantum_circuit(num_qubits, depth)

        # Classical surrogate network (mirrors EstimatorQNN example)
        self.classical_net = self._build_classical_surrogate(num_features)

        # EstimatorQNN wrapper
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.quantum_circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------ #
    # Quantum circuit construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_quantum_circuit(num_qubits: int, depth: int) -> Tuple[
        QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]
    ]:
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

        observables = [
            SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)])
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    # ------------------------------------------------------------------ #
    # Classical surrogate network (tiny FC regressor)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_classical_surrogate(num_features: int) -> nn.Module:
        """
        Build a lightweight PyTorch regression network that mimics the
        behaviour of the quantum circuit for quick CPU testing.
        """
        return nn.Sequential(
            nn.Linear(num_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    # ------------------------------------------------------------------ #
    # Prediction interface
    # ------------------------------------------------------------------ #
    def predict(self, x: Iterable[float]) -> float:
        """
        Compute the probability of class ``1`` for a single feature vector.

        Parameters
        ----------
        x : Iterable[float]
            Feature vector of length ``num_features`` (must match ``num_qubits``).

        Returns
        -------
        float
            Probability of the positive class.
        """
        import numpy as np

        # Classical surrogate prediction
        x_np = np.array(x).reshape(1, -1)
        cls_out = self.classical_net(torch.from_numpy(x_np).float()).item()

        # Bind quantum parameters
        param_bindings = dict(zip([str(p) for p in self.encoding], x))
        bound_qc = self.quantum_circuit.bind_parameters(param_bindings)

        # Evaluate expectation values
        exp_vals = self.estimator.run(
            circuit=bound_qc,
            observables=self.observables,
            parameter_values={},
        ).values

        # Combine classical and quantum outputs (simple averaging)
        prob = 0.5 * (cls_out + np.mean(exp_vals))
        return float(prob)

    def __repr__(self) -> str:
        return (
            f"HybridClassifier(num_qubits={self.num_qubits}, depth={self.depth}, "
            f"num_features={self.num_features})"
        )


__all__ = ["HybridClassifier"]
