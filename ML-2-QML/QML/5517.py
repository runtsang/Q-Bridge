"""
HybridSamplerQNN – Quantum side of the hybrid model.

This module implements:
  * a two‑qubit parameterised sampler circuit (mirrors SamplerQNN),
  * a single‑qubit estimator circuit (mirrors EstimatorQNN),
  * a wrapper that evaluates both circuits and returns probability
    distributions / expectation values.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator  # type: ignore[import]


class QuantumSampler:
    """
    Two‑qubit parameterised sampler.
    Parameters
    ----------
    input_params : ParameterVector
        Two input angles that drive Ry rotations.
    weight_params : ParameterVector
        Four weight angles that build a shallow entangling circuit.
    """

    def __init__(self) -> None:
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        self.backend = AerSimulator()
        self.compiled = transpile(self.circuit, backend=self.backend)
        self.sampler = StatevectorSampler(backend=self.backend)

    def sample(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Execute the sampler for a batch of input/weight pairs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, 2) – input angles.
        weights : np.ndarray
            Shape (batch, 4) – weight angles.

        Returns
        -------
        np.ndarray
            Shape (batch, 4) – probability distribution over 2‑qubit basis states.
        """
        batch = inputs.shape[0]
        probs = np.zeros((batch, 4))
        for i in range(batch):
            bind = {
                self.input_params[0]: inputs[i, 0],
                self.input_params[1]: inputs[i, 1],
                self.weight_params[0]: weights[i, 0],
                self.weight_params[1]: weights[i, 1],
                self.weight_params[2]: weights[i, 2],
                self.weight_params[3]: weights[i, 3],
            }
            qobj = assemble(self.compiled, parameter_binds=[bind])
            result = self.sampler.backend.run(qobj).result()
            counts = result.get_counts()
            total = sum(counts.values())
            probs[i] = [
                counts.get("00", 0),
                counts.get("01", 0),
                counts.get("10", 0),
                counts.get("11", 0),
            ] / total
        return probs


class QuantumEstimator:
    """
    Single‑qubit estimator circuit used to mimic EstimatorQNN.
    """

    def __init__(self) -> None:
        self.input_param = Parameter("input1")
        self.weight_param = Parameter("weight1")

        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        self.backend = AerSimulator()
        self.estimator = StatevectorEstimator(backend=self.backend)
        self.observable = SparsePauliOp.from_list([("Y", 1)])

    def estimate(self, input_val: float, weight_val: float) -> float:
        """
        Compute the expectation value of Y for a single input/weight pair.

        Parameters
        ----------
        input_val : float
            Input angle.
        weight_val : float
            Weight angle.

        Returns
        -------
        float
            Expected value of the observable.
        """
        bind = {self.input_param: input_val, self.weight_param: weight_val}
        result = self.estimator.run(
            self.circuit,
            parameter_binds=[bind],
            observables=[self.observable],
        )
        return float(result[0].data)


class HybridSamplerQNN:
    """
    High‑level quantum wrapper that exposes both sampler and estimator
    functionality.  It can be used as a drop‑in replacement for the
    classical SamplerQNN when a quantum backend is available.
    """

    def __init__(self) -> None:
        self.sampler = QuantumSampler()
        self.estimator = QuantumEstimator()

    def run_sampler(
        self, inputs: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Wrapper around QuantumSampler.sample().
        """
        return self.sampler.sample(inputs, weights)

    def run_estimator(
        self, input_val: float, weight_val: float
    ) -> float:
        """
        Wrapper around QuantumEstimator.estimate().
        """
        return self.estimator.estimate(input_val, weight_val)


__all__ = ["QuantumSampler", "QuantumEstimator", "HybridSamplerQNN"]
