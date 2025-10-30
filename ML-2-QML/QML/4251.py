from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class FraudDetectionHybridQuantum:
    """
    Quantum evaluator for the hybrid fraud detection model.
    Implements a simple one‑qubit variational circuit from the EstimatorQNN example.
    The circuit accepts a single input parameter and a trainable weight, returns
    the expectation value of Y.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.input_param = Parameter("input1")
        self.weight_param = Parameter("weight1")

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.input_param, 0)
        qc.rx(self.weight_param, 0)
        self.circuit = qc

        self.observable = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def evaluate(self, input_value: float, weight: float = 0.5) -> float:
        """
        Compute the expectation value for a given input value.
        The weight parameter can be tuned or learned; here a default is provided.
        """
        binding = {self.input_param: input_value, self.weight_param: weight}
        result = self.estimator_qnn.evaluate(parameters=binding)
        return result.real

    def get_quantum_fn(self) -> callable:
        """
        Return a callable that can be injected into FraudDetectionHybrid.
        """
        return lambda x: self.evaluate(x)

__all__ = ["FraudDetectionHybridQuantum"]
