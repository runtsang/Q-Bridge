"""Quantum sampler and estimator components used by the hybrid sampler network."""

from __future__ import annotations

import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

class SamplerQNN:
    """Quantum sampler wrapper exposing a.sample method."""
    def __init__(self) -> None:
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        self._qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=Sampler()
        )

    def sample(self, inputs: torch.Tensor) -> np.ndarray:
        np_inputs = inputs.detach().cpu().numpy()
        return self._qnn.sample(np_inputs)

class EstimatorQNN:
    """Quantum estimator wrapper exposing a.sample method."""
    def __init__(self) -> None:
        param1 = Parameter("input1")
        param2 = Parameter("weight1")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(param1, 0)
        qc.rx(param2, 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        self._qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[param1],
            weight_params=[param2],
            estimator=Estimator()
        )

    def sample(self, inputs: torch.Tensor) -> np.ndarray:
        np_inputs = inputs.detach().cpu().numpy()
        return self._qnn.sample(np_inputs)

__all__ = ["SamplerQNN", "EstimatorQNN"]
