"""Hybrid quantum neural network that integrates regression and sampling.

The quantum circuit is a two‑qubit parameterised circuit.  Input angles are
taken from a 2‑dimensional input vector.  The first qubit is rotated by the
regression output (treated as a weight) and the second qubit is rotated by
two sampler angles.  An expectation value of the Y operator on the first
qubit is returned by the EstimatorQNN, while a StatevectorSampler provides
the joint probability of measuring |00>, |01>, |10>, |11>.
"""

from __future__ import annotations

from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

class HybridEstimatorSampler:
    """Combines EstimatorQNN and SamplerQNN into a single quantum model."""
    def __init__(self) -> None:
        # Input parameters
        self.inputs = ParameterVector("x", 2)
        # Weight parameters: 1 for regression, 2 for sampler
        self.weight_reg = Parameter("w_reg")
        self.weights_samp = ParameterVector("w_samp", 2)

        # Build circuit template
        self.circuit = QuantumCircuit(2)
        # Input rotations
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        # Entanglement
        self.circuit.cx(0, 1)
        # Regression rotation on qubit 0
        self.circuit.ry(self.weight_reg, 0)
        # Sampler rotations on qubit 1
        self.circuit.ry(self.weights_samp[0], 1)
        self.circuit.ry(self.weights_samp[1], 1)
        # Final entanglement
        self.circuit.cx(0, 1)

        # Observables
        self.obs = [("Y" * self.circuit.num_qubits, 1)]

        # Estimator component
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.obs,
            input_params=self.inputs,
            weight_params=[self.weight_reg],
            estimator=self.estimator,
        )

        # Sampler component
        self.sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights_samp,
            sampler=self.sampler,
        )

    def set_weights(self, reg: float, samp: list[float]) -> None:
        """Assign runtime weight values to the quantum circuit."""
        bound = {self.weight_reg: reg}
        bound.update({self.weights_samp[i]: samp[i] for i in range(2)})
        self.circuit.assign_parameters(bound, inplace=True)

    def get_estimator(self):
        """Return the EstimatorQNN instance."""
        return self.estimator_qnn

    def get_sampler(self):
        """Return the SamplerQNN instance."""
        return self.sampler_qnn

__all__ = ["HybridEstimatorSampler"]
