"""Hybrid quantum sampler‑estimator.

The class encapsulates a Qiskit sampler circuit and a Qiskit estimator
circuit. It exposes two methods:

    sample(params: torch.Tensor) -> torch.Tensor
        returns probability distribution

    estimate(params: torch.Tensor) -> torch.Tensor
        returns expectation value

The parameters are expected to be the outputs of the classical
SamplerQNN network defined in the ML module: the first six elements
are for the sampler circuit (2 input, 4 weight) and the last two
elements are for the estimator circuit (input, weight).
"""

from __future__ import annotations

from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
import torch
import numpy as np


class SamplerQNN:
    """Hybrid quantum sampler‑estimator.

    The class builds two Qiskit circuits: a sampler circuit that
    produces a probability distribution over two outcomes, and an
    estimator circuit that returns the expectation value of a Pauli-Y
    observable on a single qubit. The circuits are wrapped by the
    corresponding Qiskit Machine Learning neural‑network classes.
    """

    def __init__(self) -> None:
        # --- Sampler circuit -------------------------------------------------
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)

        sampler_primitive = StatevectorSampler()
        self.sampler = QSamplerQNN(
            circuit=qc2,
            input_params=inputs2,
            weight_params=weights2,
            sampler=sampler_primitive,
        )

        # --- Estimator circuit -----------------------------------------------
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)

        observable1 = SparsePauliOp.from_list([("Y", 1)])

        estimator_primitive = StatevectorEstimator()
        self.estimator = QEstimatorQNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=estimator_primitive,
        )

    def sample(self, sampler_params: torch.Tensor) -> torch.Tensor:
        """Execute the sampler circuit with the given parameters.

        sampler_params: Tensor of shape (..., 6). The first two elements
        are the input parameters, the next four are the weight parameters.
        """
        inputs = sampler_params[..., :2]
        weights = sampler_params[..., 2:6]
        probs = []
        for inp, w in zip(inputs.cpu().numpy(), weights.cpu().numpy()):
            probs.append(self.sampler(inputs=inp, weights=w))
        return torch.tensor(probs)

    def estimate(self, estimator_params: torch.Tensor) -> torch.Tensor:
        """Execute the estimator circuit with the given parameters.

        estimator_params: Tensor of shape (..., 2). The first element
        is the input parameter, the second is the weight parameter.
        """
        inputs = estimator_params[..., :1]
        weights = estimator_params[..., 1:2]
        expectations = []
        for inp, w in zip(inputs.cpu().numpy(), weights.cpu().numpy()):
            expectations.append(self.estimator(inputs=inp, weights=w))
        return torch.tensor(expectations)


__all__ = ["SamplerQNN"]
