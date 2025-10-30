"""Hybrid quantum sampler and estimator.

This module builds a single parameterised quantum circuit that can be used
with both the Qiskit `SamplerQNN` and `EstimatorQNN` wrappers.  The circuit
uses two qubits, Ry rotations for the two input parameters and a sequence
of CNOTs and Ry weight rotations.  The sampler wrapper samples the
computational basis distribution, while the estimator wrapper measures
the expectation value of a Y observable on the first qubit.  By exposing
both wrappers in a single class, experiments can jointly optimise
sampling probabilities and expectation values.
"""

from __future__ import annotations

from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class HybridQNN:
    """Hybrid quantum sampler / estimator wrapper.

    Attributes
    ----------
    sampler_qnn : qiskit_machine_learning.neural_networks.SamplerQNN
        Sampler wrapper around the shared circuit.
    estimator_qnn : qiskit_machine_learning.neural_networks.EstimatorQNN
        Estimator wrapper around the same circuit.
    """

    def __init__(self) -> None:
        # Parameter vectors for inputs and weights
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build a shared circuit
        qc = QuantumCircuit(2)
        # Input rotations
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        # Entangling layer
        qc.cx(0, 1)
        # Weight rotations
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)

        # Observable for estimation: Y on first qubit
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

        # Sampler wrapper
        sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=sampler,
        )

        # Estimator wrapper
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.inputs[0]],
            weight_params=[self.weights[1]],
            estimator=estimator,
        )

    def sample(self, input_vals: list[float], weight_vals: list[float], n_shots: int = 1024):
        """
        Sample from the quantum circuit using the sampler wrapper.

        Parameters
        ----------
        input_vals : list[float]
            Values for the two input parameters.
        weight_vals : list[float]
            Values for the four weight parameters.
        n_shots : int, optional
            Number of shots to draw.

        Returns
        -------
        dict
            Frequency dictionary of measurement outcomes.
        """
        param_dict = {
            str(self.inputs[0]): input_vals[0],
            str(self.inputs[1]): input_vals[1],
            str(self.weights[0]): weight_vals[0],
            str(self.weights[1]): weight_vals[1],
            str(self.weights[2]): weight_vals[2],
            str(self.weights[3]): weight_vals[3],
        }
        return self.sampler_qnn.sample(param_dict, shots=n_shots)

    def estimate(self, input_val: float, weight_val: float):
        """
        Estimate expectation value of the Y observable using the estimator wrapper.

        Parameters
        ----------
        input_val : float
            Value for the first input parameter.
        weight_val : float
            Value for the second weight parameter.

        Returns
        -------
        float
            Estimated expectation value.
        """
        param_dict = {
            str(self.inputs[0]): input_val,
            str(self.weights[1]): weight_val,
        }
        return self.estimator_qnn.predict(param_dict)[0]

__all__ = ["HybridQNN"]
