"""Extended quantum estimator with a variational circuit.

The QML version uses Qiskit to construct a multi‑qubit variational circuit
with parametrised Ry/Rz and entangling CX layers.  It wraps the circuit
in Qiskit Machine Learning's EstimatorQNN and exposes a `predict` method
that evaluates the expectation value of a multi‑qubit Z observable for
arbitrary classical input vectors.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from typing import Sequence

class EstimatorQNNExtended:
    """
    Variational quantum neural network with configurable depth.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    layers : int, default 2
        Number of parameterised layers (each contains Ry and Rz on every qubit).
    init_weights : np.ndarray | None, default None
        Flat array to initialise weight parameters.
    """
    def __init__(
        self,
        num_qubits: int = 2,
        layers: int = 2,
        init_weights: np.ndarray | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers

        # Parameters for inputs and weights
        self.input_params: list[Parameter] = [
            Parameter(f"inp_{i}") for i in range(num_qubits)
        ]
        self.weight_params: list[Parameter] = [
            Parameter(f"w_{i}") for i in range(num_qubits * layers)
        ]

        # Build the parameterised circuit
        self.circuit = self._build_circuit()

        # Observable: total Z on all qubits (can be changed if desired)
        self.observables = SparsePauliOp.from_list([(Pauli("Z" * num_qubits), 1.0)])

        # Estimator primitive for expectation values
        self.estimator = Estimator()

        # Wrap into EstimatorQNN
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

        if init_weights is not None:
            self.set_weights(init_weights)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for l in range(self.layers):
            for i in range(self.num_qubits):
                qc.ry(self.input_params[i], i)
                qc.rz(self.weight_params[l * self.num_qubits + i], i)
            # Entangling CX chain
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def set_weights(self, weights: np.ndarray) -> None:
        """Assign a flat array of weight values to the circuit parameters."""
        if len(weights)!= len(self.weight_params):
            raise ValueError("Weight array length mismatch.")
        param_dict = {p: w for p, w in zip(self.weight_params, weights)}
        self.circuit.assign_parameters(param_dict, inplace=True)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute expectation values for a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray, shape (batch, num_qubits)
            Classical feature vectors.
        Returns
        -------
        np.ndarray
            Predicted outputs of shape (batch,).
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        results = []
        for inp in inputs:
            param_dict = {p: v for p, v in zip(self.input_params, inp)}
            exp_val = self.estimator_qnn.predict(param_dict)[0]
            results.append(exp_val)
        return np.array(results)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.predict(inputs)

__all__ = ["EstimatorQNNExtended"]
