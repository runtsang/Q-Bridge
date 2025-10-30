"""Quantum estimator that mirrors the classical architecture using a variational circuit.

The module builds a parameterised quantum circuit that encodes 2×2 image patches
into a 4‑qubit register, applies a random layer, and finally a single weight‑parameter
rotation before measuring in the Pauli‑Z basis.  An observable that measures the
Y‑Pauli string over all qubits is used to compute the expectation value, which
serves as the network output.  The qiskit‑machine‑learning `EstimatorQNN` class is
leveraged to handle the forward pass and optimisation of both input and weight
parameters.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QuantumFeatureMap:
    """Encodes a 2×2 image patch into a 4‑qubit state via Ry rotations."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.params = [Parameter(f"phi_{i}") for i in range(n_qubits)]
        self.circuit = QuantumCircuit(n_qubits)
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        self.circuit.barrier()

    def bind(self, data: np.ndarray) -> QuantumCircuit:
        """Return a circuit with the feature parameters bound to the provided data."""
        return self.circuit.bind_parameters(
            {p: v for p, v in zip(self.params, data)}
        )


class EstimatorQNN_QML:
    """Quantum neural network that evaluates a variational circuit for regression."""
    def __init__(self) -> None:
        # Feature map and weight parameters
        self.feature_map = QuantumFeatureMap()
        self.weight_param = Parameter("w")

        # Build the full circuit: feature map + single rotation + measurement
        self.circuit = QuantumCircuit(self.feature_map.n_qubits)
        self.circuit.append(self.feature_map.circuit, range(self.feature_map.n_qubits))
        self.circuit.rx(self.weight_param, 0)

        # Observable: Y Pauli string over all qubits
        self.observable = SparsePauliOp.from_list(
            [(("Y" * self.feature_map.n_qubits), 1)]
        )

        # Estimator primitive
        self.estimator = StatevectorEstimator()

        # Wrap in qiskit‑ml EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.feature_map.params,
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the quantum network on a batch of 2×2 patch data.

        Parameters
        ----------
        data : np.ndarray
            Shape (batch, n_qubits) where each row contains the 4 values of a
            2×2 image patch.  The values are expected to be in the range [0, π]
            to match the Ry encoding.

        Returns
        -------
        np.ndarray
            The network output for each batch element.
        """
        return self.qnn(data)


__all__ = ["EstimatorQNN_QML"]
