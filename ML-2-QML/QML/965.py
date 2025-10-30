"""
Enhanced variational quantum neural network.

The circuit operates on two qubits and consists of:
  * An input encoding layer that applies RX rotations to each qubit.
  * Two entanglement layers (CX gates) to enable correlations.
  * Two parameterised rotation layers (RY, RZ) that serve as trainable weights.
  * Measurement of the Pauli‑Z observable on both qubits.

The QNN is built on top of Qiskit Machine Learning's EstimatorQNN and
leverages the StatevectorEstimator for exact gradients, making it suitable
for research experiments that require high precision.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from typing import List, Tuple


class EstimatorQNNEnhanced:
    """
    Variational QNN with two qubits, entanglement and multiple rotation layers.

    Parameters
    ----------
    num_layers : int
        Number of rotation layers (default 2).  Each layer adds a set of
        parameterised RY and RZ gates per qubit.
    """

    def __init__(self, num_layers: int = 2) -> None:
        self.num_layers = num_layers
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self._build_observable(),
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
        """
        Construct the variational circuit.

        Returns
        -------
        tuple
            (QuantumCircuit, list of input Parameters, list of weight Parameters)
        """
        qc = QuantumCircuit(2)

        # Input encoding: RX rotations
        in_params = [Parameter(f"inp_{i}") for i in range(2)]
        for i, p in enumerate(in_params):
            qc.rx(p, i)

        # Entanglement layer
        qc.cx(0, 1)

        # Parameterised rotation layers
        weight_params = []
        for layer in range(self.num_layers):
            layer_params = [Parameter(f"w_{layer}_{i}") for i in range(2)]
            weight_params.extend(layer_params)
            for i, p in enumerate(layer_params):
                qc.ry(p, i)
                qc.rz(p, i)
            # Additional entanglement
            qc.cx(0, 1)

        return qc, in_params, weight_params

    def _build_observable(self) -> SparsePauliOp:
        """
        Observable is the sum of Pauli‑Z on both qubits.

        Returns
        -------
        SparsePauliOp
            Observable operator.
        """
        return SparsePauliOp.from_list([("Z" * self.circuit.num_qubits, 1)])

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward evaluation of the QNN.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch_size, 2) with input features.

        Returns
        -------
        np.ndarray
            Predicted outputs of shape (batch_size, 1).
        """
        return np.array([self.qnn.predict(inputs[i, :])[0] for i in range(inputs.shape[0])])

    def get_parameters(self) -> np.ndarray:
        """
        Return current weight parameters.

        Returns
        -------
        np.ndarray
            Flattened array of all trainable parameters.
        """
        return np.array(self.qnn.parameters)

    def set_parameters(self, params: np.ndarray) -> None:
        """
        Update the QNN with new parameters.

        Parameters
        ----------
        params : np.ndarray
            Flattened array of parameters matching the QNN's weight space.
        """
        self.qnn.parameters = params

__all__ = ["EstimatorQNNEnhanced"]
