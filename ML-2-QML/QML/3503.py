"""Hybrid quantum neural network providing both regression and sampling outputs."""
from __future__ import annotations

from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
import numpy as np

class HybridQNN:
    """
    Quantum hybrid network that combines a regression EstimatorQNN
    and a sampling SamplerQNN.  It exposes an evaluate method that
    returns both the expectation value (regression) and the sample
    distribution (generative) for the same input.
    """

    def __init__(self) -> None:
        # Regression circuit (mirrors EstimatorQNN seed)
        reg_params = [Parameter("input1"), Parameter("weight1")]
        reg_circuit = QuantumCircuit(1)
        reg_circuit.h(0)
        reg_circuit.ry(reg_params[0], 0)
        reg_circuit.rx(reg_params[1], 0)

        reg_observable = SparsePauliOp.from_list([("Y", 1)])

        self._estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=reg_circuit,
            observables=reg_observable,
            input_params=[reg_params[0]],
            weight_params=[reg_params[1]],
            estimator=self._estimator,
        )

        # Sampling circuit (mirrors SamplerQNN seed)
        samp_inputs = ParameterVector("input", length=2)
        samp_weights = ParameterVector("weight", length=4)

        samp_circuit = QuantumCircuit(2)
        samp_circuit.ry(samp_inputs[0], 0)
        samp_circuit.ry(samp_inputs[1], 1)
        samp_circuit.cx(0, 1)
        samp_circuit.ry(samp_weights[0], 0)
        samp_circuit.ry(samp_weights[1], 1)
        samp_circuit.cx(0, 1)
        samp_circuit.ry(samp_weights[2], 0)
        samp_circuit.ry(samp_weights[3], 1)

        self._sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=samp_circuit,
            input_params=samp_inputs,
            weight_params=samp_weights,
            sampler=self._sampler,
        )

    def evaluate(self, inputs: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluate both regression and sampling components.

        Parameters
        ----------
        inputs : np.ndarray
            Input array of shape (2,) for regression and (2,) for sampler.
        weights : np.ndarray
            Weight array of shape (1,) for regression and (4,) for sampler.

        Returns
        -------
        tuple[float, np.ndarray]
            The expectation value from the regression EstimatorQNN and the
            probability distribution from the SamplerQNN.
        """
        # Regression expectation value
        reg_exp = self.estimator_qnn.predict(inputs, weights)

        # Sampling distribution
        samp_probs = self.sampler_qnn.sample(inputs, weights)

        return reg_exp, samp_probs

    def parameters(self) -> list[Parameter]:
        """
        Return all quantum parameters (input and weight) for both sub-networks.
        """
        return (
            list(self.estimator_qnn.input_params)
            + list(self.estimator_qnn.weight_params)
            + list(self.sampler_qnn.input_params)
            + list(self.sampler_qnn.weight_params)
        )

__all__ = ["HybridQNN"]
