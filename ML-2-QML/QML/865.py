"""Variational quantum neural network for regression.

This implementation builds a multi‑qubit parameterised circuit
with configurable depth and qubits.  It uses Qiskit’s
StatevectorEstimator for expectation evaluation and exposes a
simple ``forward`` method that accepts a single input vector.
The API mirrors the original EstimatorQNN example by providing
a factory function ``EstimatorQNN`` that returns an instance.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from typing import List, Tuple


class EstimatorQNNModel:
    """Variational quantum neural network with configurable depth and qubits."""

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 3,
        observable: SparsePauliOp | None = None,
        estimator: StatevectorEstimator | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit = self._build_circuit()
        self.input_params, self.weight_params = self._extract_params()

        if observable is None:
            # Default observable: Pauli‑Y on the first qubit
            observable = SparsePauliOp.from_list([("Y" + "I" * (num_qubits - 1), 1)])
        self.observable = observable

        if estimator is None:
            estimator = StatevectorEstimator()
        self.estimator = estimator

        # Wrap into Qiskit’s EstimatorQNN for convenience
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Initial layer of Hadamards
        for q in range(self.num_qubits):
            qc.h(q)

        # Variational layers
        for d in range(self.depth):
            for q in range(self.num_qubits):
                theta = Parameter(f"theta_{d}_{q}")
                qc.ry(theta, q)
            # Entangling layer (full‑connectivity)
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.num_qubits - 1, 0)
        return qc

    def _extract_params(self) -> Tuple[List[Parameter], List[Parameter]]:
        input_params: List[Parameter] = []
        weight_params: List[Parameter] = []
        for p in self.circuit.parameters:
            if "theta" in p.name:
                weight_params.append(p)
            else:
                input_params.append(p)
        return input_params, weight_params

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit expectation for a single input vector.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (num_inputs,) where ``num_inputs`` equals
            ``len(self.input_params)``.  The values are mapped to the
            circuit’s input parameters.

        Returns
        -------
        np.ndarray
            Expectation value of the observable.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        param_dict = {p: val for p, val in zip(self.input_params, inputs[0])}
        result = self.estimator.run(
            circuits=[self.circuit],
            parameter_values=[param_dict],
        )[0]
        return np.array(result.values[0])

def EstimatorQNN() -> EstimatorQNNModel:
    """Factory that returns a ready‑to‑use quantum estimator."""
    return EstimatorQNNModel()


__all__ = ["EstimatorQNN"]
