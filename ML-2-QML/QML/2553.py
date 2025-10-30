from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
import numpy as np

class EstimatorQNN:
    """Quantum neural network that encodes a single input value into a single qubit
    via a parameterized rotation and measures the Y‑observable.  The network
    exposes a `predict` method returning the expectation value for each sample."""
    def __init__(self, weight_init: float = 0.0):
        # Parameters
        self.input_param = Parameter("x")
        self.weight_param = Parameter("w")

        # Circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        # Observable (Pauli Y)
        self.observable = np.array([[0, -1j], [1j, 0]])

        # Estimator backend
        self.estimator = StatevectorEstimator()

        # Wrap into Qiskit’s EstimatorQNN
        self.qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the expectation value for each input sample."""
        return np.array([self.qnn.predict({self.input_param: xi})[0] for xi in x])

def EstimatorQNN() -> EstimatorQNN:
    """Return an instance of the quantum EstimatorQNN."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN"]
