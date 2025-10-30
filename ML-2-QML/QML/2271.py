"""HybridSamplerEstimatorQNN: Quantum hybrid sampler‑estimator network.

This module implements a variational quantum circuit that
simultaneously produces a probability distribution (for sampling)
and an expectation value (for regression).  The circuit is
parameterised by two sets of parameters: input parameters that
encode the classical data and weight parameters that are trained.
Both a SamplerQNN and an EstimatorQNN are instantiated from the
same circuit, sharing entanglement and rotation layers, which
enables joint optimisation in a quantum‑classical hybrid setting.

The design follows the combination scaling paradigm: the quantum
branch is a single variational circuit that outputs both
probabilities and expectation values, while the classical
branch (defined in the companion ML module) mirrors the same
two‑head architecture.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator


class HybridSamplerEstimatorQNN:
    """Quantum hybrid sampler‑estimator network."""

    def __init__(self) -> None:
        # Parameter vectors for data encoding and trainable weights
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Build a shared variational circuit
        self.circuit = QuantumCircuit(2)
        # Data encoding: individual RY rotations
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        # Entanglement
        self.circuit.cx(0, 1)
        # Trainable rotation layers
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Sampler QNN: returns probability distribution
        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
        )

        # Estimator QNN: returns expectation of Y on qubit 0
        observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=[self.input_params[0]],
            weight_params=[self.weight_params[0]],
            estimator=estimator,
        )

    def sample(self, input_vals: np.ndarray, weight_vals: np.ndarray) -> np.ndarray:
        """
        Execute the sampler branch.

        Parameters
        ----------
        input_vals : np.ndarray
            Array of shape (batch, 2) with input data.
        weight_vals : np.ndarray
            Array of shape (batch, 4) with weight parameters.

        Returns
        -------
        np.ndarray
            Sampling probabilities of shape (batch, 2).
        """
        return self.sampler_qnn.predict(input_vals, weight_vals)

    def estimate(self, input_val: float, weight_val: float) -> float:
        """
        Execute the estimator branch.

        Parameters
        ----------
        input_val : float
            Single input value for the first qubit.
        weight_val : float
            Single weight value for the first rotation.

        Returns
        -------
        float
            Expectation value of the observable.
        """
        return self.estimator_qnn.predict(input_val, weight_val)


__all__ = ["HybridSamplerEstimatorQNN"]
