"""Hybrid estimator implemented with Qiskit.

The class builds a parameterized quantum circuit that first encodes two
classical input features as rotation angles on a single qubit, then applies
a small variational block with trainable weights. It returns an EstimatorQNN
instance that can be used with Qiskit's StatevectorEstimator or any other
primitives. The observable is a Pauli‑Y operator, enabling a richer
expectation value for regression tasks.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


class HybridEstimatorQNN:
    """
    Qiskit implementation of a hybrid estimator.

    Parameters
    ----------
    input_dim : int
        Number of classical input features to encode.
    weight_dim : int
        Number of trainable weight parameters for the variational block.
    """
    def __init__(self, input_dim: int = 2, weight_dim: int = 2) -> None:
        self.input_dim = input_dim
        self.weight_dim = weight_dim

        # Define parameters
        self.input_params = [Parameter(f"x{i}") for i in range(self.input_dim)]
        self.weight_params = [Parameter(f"w{i}") for i in range(self.weight_dim)]

        # Build the circuit
        self.circuit = self._build_circuit()

        # Observable
        self.observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        # Encode inputs as rotations
        for i, param in enumerate(self.input_params):
            qc.ry(param, 0)
        # Variational block
        for param in self.weight_params:
            qc.rz(param, 0)
            qc.rx(param, 0)
        qc.h(0)
        return qc

    def get_estimator_qnn(self) -> QiskitEstimatorQNN:
        """Return a Qiskit EstimatorQNN wrapping the circuit."""
        return QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def __call__(self, inputs: list[list[float]]) -> list[float]:
        """Convenience wrapper to evaluate the estimator on a batch of inputs."""
        return self.get_estimator_qnn().partial_fit(inputs, [0.0] * len(inputs))[0]


__all__ = ["HybridEstimatorQNN"]
