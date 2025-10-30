"""Hybrid quantum sampler‑estimator neural network.

The class builds two parameterised quantum circuits:
- a 2‑qubit sampler circuit that outputs a probability distribution
  over two basis states;
- a 1‑qubit estimator circuit that measures the expectation of Y.
Both circuits are wrapped with Qiskit Machine Learning QNNs
(`SamplerQNN` and `EstimatorQNN`).  The class exposes a `run`
method that accepts classical input values and weight values,
updates the circuit parameters, and returns the sampler
probabilities and the estimator expectation value.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN

import numpy as np


class HybridQNN:
    """Quantum hybrid sampler‑estimator neural network."""

    def __init__(self) -> None:
        # ---- Sampler circuit (2 qubits) ----
        self.sampler_inputs = ParameterVector("inp_s", 2)
        self.sampler_weights = ParameterVector("w_s", 4)
        qc_sampler = QuantumCircuit(2)
        qc_sampler.ry(self.sampler_inputs[0], 0)
        qc_sampler.ry(self.sampler_inputs[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(self.sampler_weights[0], 0)
        qc_sampler.ry(self.sampler_weights[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(self.sampler_weights[2], 0)
        qc_sampler.ry(self.sampler_weights[3], 1)

        sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=qc_sampler,
            input_params=self.sampler_inputs,
            weight_params=self.sampler_weights,
            sampler=sampler,
        )

        # ---- Estimator circuit (1 qubit) ----
        self.estimator_inputs = Parameter("inp_e")
        self.estimator_weights = Parameter("w_e")
        qc_estimator = QuantumCircuit(1)
        qc_estimator.h(0)
        qc_estimator.ry(self.estimator_inputs, 0)
        qc_estimator.rx(self.estimator_weights, 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = QEstimatorQNN(
            circuit=qc_estimator,
            observables=observable,
            input_params=[self.estimator_inputs],
            weight_params=[self.estimator_weights],
            estimator=estimator,
        )

    def run(
        self,
        sampler_input: np.ndarray,
        sampler_weight: np.ndarray,
        estimator_input: float,
        estimator_weight: float,
    ) -> tuple[np.ndarray, float]:
        """Evaluate both QNNs with the supplied parameters.

        Parameters
        ----------
        sampler_input : array_like, shape (2,)
            Classical input values for the sampler circuit.
        sampler_weight : array_like, shape (4,)
            Weight parameters for the sampler circuit.
        estimator_input : float
            Classical input value for the estimator circuit.
        estimator_weight : float
            Weight parameter for the estimator circuit.

        Returns
        -------
        probs : np.ndarray, shape (2,)
            Sampler probability distribution.
        exp_val : float
            Expectation value of the Y observable from the estimator.
        """
        # Execute sampler
        probs = self.sampler_qnn.forward(
            input_params=[sampler_input],
            weight_params=[sampler_weight],
        )

        # Execute estimator
        exp_val = self.estimator_qnn.forward(
            input_params=[estimator_input],
            weight_params=[estimator_weight],
        )

        return probs, float(exp_val)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sampler_qnn={self.sampler_qnn}, estimator_qnn={self.estimator_qnn})"
