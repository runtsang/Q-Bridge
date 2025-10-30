"""Hybrid quantum sampler and estimator network.

The circuit operates on two qubits.  It first applies input rotations,
then a controlled‑X entangling layer, followed by two layers of
parameterized rotations that act as trainable weights.  The same
circuit is used by the qiskit_machine_learning.neural_networks
SamplerQNN and EstimatorQNN objects.  A wrapper class exposes a
`forward` method returning a probability distribution (via the
sampler) and an expectation value (via the estimator).
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


class HybridSamplerEstimatorQNN:
    """
    Quantum hybrid network combining a sampler and an estimator.

    The underlying circuit has two qubits and uses four trainable weight
    parameters.  The input parameters control the initial rotations,
    while the weight parameters are shared between the sampler and
    estimator heads.

    The `forward` method returns a tuple (probs, value) where `probs`
    is a probability distribution over two measurement outcomes and
    `value` is the expectation value of the Y Pauli operator on the
    first qubit.
    """

    def __init__(self) -> None:
        # Define parameters
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Build the circuit
        self.circuit = QuantumCircuit(2)
        # Input rotations
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        # Entanglement
        self.circuit.cx(0, 1)
        # Weight rotations
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Sampler and Estimator primitives
        self.sampler_primitive = StatevectorSampler()
        self.estimator_primitive = StatevectorEstimator()

        # Sampler QNN
        self.sampler_qnn = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler_primitive,
        )

        # Estimator QNN
        observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])
        self.estimator_qnn = QEstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=[self.input_params[0]],
            weight_params=[self.weight_params[0]],
            estimator=self.estimator_primitive,
        )

    def forward(self, input_vals: list[float], weight_vals: list[float]) -> tuple[list[float], float]:
        """
        Evaluate the hybrid QNN.

        Parameters
        ----------
        input_vals : list[float]
            Numerical values for the two input parameters.
        weight_vals : list[float]
            Numerical values for the four weight parameters.

        Returns
        -------
        probs : list[float]
            Probabilities of measuring |00> and |01> (first qubit in 0 or 1).
        value : float
            Expectation value of the observable used by the estimator.
        """
        # Bind parameters
        bound_circuit = self.circuit.bind_parameters(
            {
                self.input_params[0]: input_vals[0],
                self.input_params[1]: input_vals[1],
                self.weight_params[0]: weight_vals[0],
                self.weight_params[1]: weight_vals[1],
                self.weight_params[2]: weight_vals[2],
                self.weight_params[3]: weight_vals[3],
            }
        )

        # Sample probabilities
        probs = self.sampler_qnn(bound_circuit).tolist()
        # Estimate expectation
        value = self.estimator_qnn(bound_circuit).item()

        return probs, value


__all__ = ["HybridSamplerEstimatorQNN"]
