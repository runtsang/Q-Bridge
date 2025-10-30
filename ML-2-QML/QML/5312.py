import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Estimator

class HybridFCL:
    """Quantum counterpart of the classical HybridFCL.  The circuit combines:

      * a Z‑feature map,
      * a layered convolution‑pooling ansatz,
      * Pauli‑Z measurements on each qubit.

    The public API matches the classical version:
        - ``run(thetas)`` evaluates the circuit for a single set of
          variational parameters and returns the expectation vector.
    """

    def __init__(self, num_qubits: int = 4, depth: int = 2) -> None:
        algorithm_globals.random_seed = 12345
        self.num_qubits = num_qubits
        self.depth = depth

        # Feature map – ZFeatureMap
        self.feature_map = ZFeatureMap(num_qubits)

        # Build ansatz
        self.ansatz, self.input_params, self.weight_params = self._build_ansatz()

        # Observables – Pauli‑Z on each qubit
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        # Estimator for evaluating the circuit
        self.estimator = Estimator()

        # Full QNN
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_ansatz(self):
        """Construct a layered ansatz with convolution and pooling operators."""
        qc = QuantumCircuit(self.num_qubits)
        # Encode feature map
        qc.append(self.feature_map, range(self.num_qubits))

        # Variational parameters
        total_var = self.num_qubits * self.depth + self.num_qubits // 2
        weight_params = ParameterVector("theta", total_var)
        conv_params = weight_params[: self.num_qubits * self.depth]
        pool_params = weight_params[self.num_qubits * self.depth :]

        # Convolutional layers
        for i in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(conv_params[i * self.num_qubits + q], q)
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)

        # Pooling layers
        for q, p in zip(range(self.num_qubits // 2), pool_params):
            qc.cx(q, self.num_qubits - q - 1)
            qc.rz(p, self.num_qubits - q - 1)

        return qc, self.feature_map.parameters, weight_params

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the QNN for a batch of feature vectors.

        Args:
            x: np.ndarray of shape (batch, num_qubits)
        Returns:
            np.ndarray of shape (batch, num_qubits) – expectation values.
        """
        return self.qnn.predict(x)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper: ``thetas`` is the flattened weight vector
        for the variational parameters.  The method returns the expectation
        values for a single input of zeros (the feature map is inactive).
        """
        dummy_input = np.zeros((1, self.num_qubits))
        weight_bindings = {p: theta for p, theta in zip(self.weight_params, thetas)}
        return self.qnn.predict(dummy_input, weight_bindings)

    def num_params(self) -> int:
        """Return total number of trainable parameters."""
        return len(self.weight_params)

    def weight_sizes(self) -> list[int]:
        """Return list of parameter counts per layer (convolution, pooling)."""
        conv_params = self.depth * self.num_qubits
        pool_params = self.num_qubits // 2
        return [conv_params, pool_params]

    def get_observables(self) -> list[SparsePauliOp]:
        """Return the observable list used in the QNN."""
        return self.observables

def FCL() -> HybridFCL:
    """Factory returning a quantum HybridFCL instance."""
    return HybridFCL()

__all__ = ["HybridFCL", "FCL"]
