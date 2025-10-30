"""Quantum hybrid estimator for EstimatorQNNGen104.

This module builds a Qiskit EstimatorQNN that mirrors the classical
architecture.  It accepts the same feature vector as the PyTorch model
and produces an expectation value that can be used as a regression
output.  The quantum circuit is kept deliberately simple to keep the
simulation cost low while still providing a non‑linear feature map.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

__all__ = ["EstimatorQNNGen104Quantum"]


class EstimatorQNNGen104Quantum:
    """Hybrid quantum estimator that emulates the classical architecture.

    Parameters
    ----------
    use_conv : bool
        Include a parameter that corresponds to the ConvFilter output.
    use_sampler : bool
        Include a parameter that corresponds to the SamplerModule output.
    """

    def __init__(self, use_conv: bool = True, use_sampler: bool = True) -> None:
        self.use_conv = use_conv
        self.use_sampler = use_sampler

        # Build a 1‑qubit circuit with several parameterized rotations
        qc = QuantumCircuit(1)
        self.input_params = [Parameter("input1"), Parameter("weight1")]
        self.weight_params = [Parameter("weight1")]

        qc.h(0)
        qc.ry(self.input_params[0], 0)
        qc.rx(self.weight_params[0], 0)

        if self.use_conv:
            self.input_params.append(Parameter("conv"))
            qc.ry(self.input_params[2], 0)

        if self.use_sampler:
            self.input_params.append(Parameter("sampler"))
            qc.ry(self.input_params[3], 0)

        self.circuit = qc
        self.observable = SparsePauliOp.from_list([("Z", 1)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(
        self,
        inputs: np.ndarray,
        conv_feature: float | None = None,
        sampler_feature: float | None = None,
    ) -> float:
        """
        Execute the quantum circuit with bound parameters.

        Parameters
        ----------
        inputs : array‑like, shape (2,)
            Base input features: [input1, weight1].
        conv_feature : float, optional
            Value produced by the ConvFilter.
        sampler_feature : float, optional
            Value produced by the SamplerModule.

        Returns
        -------
        float
            Expectation value of the observable Z.
        """
        binds = {
            self.input_params[0]: float(inputs[0]),
            self.weight_params[0]: float(inputs[1]),
        }

        idx = 1
        if self.use_conv:
            if conv_feature is None:
                raise ValueError("conv_feature required when use_conv is True")
            binds[self.input_params[idx]] = float(conv_feature)
            idx += 1

        if self.use_sampler:
            if sampler_feature is None:
                raise ValueError("sampler_feature required when use_sampler is True")
            binds[self.input_params[idx]] = float(sampler_feature)

        expectation = self.qnn.run(bindings=binds)[0]
        return float(expectation)
