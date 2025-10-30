"""Quantum hybrid sampler–estimator using Qiskit."""
from __future__ import annotations

from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

class HybridSamplerEstimatorQNN:
    """Quantum circuit that implements both a sampler and an estimator.

    The circuit uses three qubits: the first two form the sampler
    subsystem and the third qubit hosts the estimator observable.
    Input parameters control rotation angles on the sampler qubits,
    while weight parameters control both sampler and estimator layers.
    The class exposes two Qiskit Machine Learning wrappers providing
    a variational sampler and a variational estimator that share
    the same underlying circuit.
    """
    def __init__(self) -> None:
        # Parameter vectors
        self.inputs_sampler = ParameterVector("input_s", 2)
        self.weights_sampler = ParameterVector("weight_s", 4)
        self.input_estimator = Parameter("input_e")
        self.weight_estimator = Parameter("weight_e")

        # Construct circuit
        self.circuit = QuantumCircuit(3)
        # Sampler part
        self.circuit.ry(self.inputs_sampler[0], 0)
        self.circuit.ry(self.inputs_sampler[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights_sampler[0], 0)
        self.circuit.ry(self.weights_sampler[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights_sampler[2], 0)
        self.circuit.ry(self.weights_sampler[3], 1)
        # Estimator part (on qubit 2)
        self.circuit.h(2)
        self.circuit.ry(self.input_estimator, 2)
        self.circuit.rx(self.weight_estimator, 2)

        # Observable for estimator
        self.observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])

        # Sampler wrapper
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs_sampler,
            weight_params=self.weights_sampler,
            sampler=StatevectorSampler()
        )

        # Estimator wrapper
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_estimator],
            weight_params=[self.weight_estimator],
            estimator=StatevectorEstimator()
        )

    def get_sampler(self) -> SamplerQNN:
        """Return the variational sampler."""
        return self.sampler_qnn

    def get_estimator(self) -> EstimatorQNN:
        """Return the variational estimator."""
        return self.estimator_qnn

__all__ = ["HybridSamplerEstimatorQNN"]
