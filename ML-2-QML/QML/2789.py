from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

class EstimatorQNN:
    """
    Quantum hybrid estimator‑sampler.

    The underlying circuit has:
        * 2 input parameters (x0, x1)
        * 4 weight parameters (w0..w3)
    Regression output: expectation value of Y⊗I.
    Sampling output: distribution over |00>, |01>, |10>, |11> after measurement.
    """
    def __init__(self) -> None:
        # Define parameters
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)

        # Build the parameterized circuit
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        # Estimator component
        observable = SparsePauliOp.from_list([("Y", 1), ("I", 1)])  # Y⊗I
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=inputs,
            weight_params=weights,
            estimator=estimator
        )

        # Sampler component
        sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler
        )

    def predict(self, inputs: list | tuple) -> float:
        """
        Return the regression expectation value.
        Accepts a list/tuple of two floats.
        """
        return self.estimator_qnn.predict(inputs)

    def sample(self, inputs: list | tuple) -> list:
        """
        Return a probability distribution over two outcomes.
        The output is a list of two floats summing to 1.
        """
        return self.sampler_qnn.sample(inputs)

__all__ = ["EstimatorQNN"]
