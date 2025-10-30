from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit import Aer
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np


class EstimatorQNNGen208:
    """
    Variational quantum circuit that mirrors the classical EstimatorQNNGen208.
    The circuit operates on *num_qubits* qubits and accepts *input_params*
    (features) and *weight_params* (trainable angles).  The observables are
    a tensor‑product of Pauli‑Y on each qubit, which makes the expectation
    value sensitive to entanglement introduced by the circuit.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        input_dim: int = 2,
        weight_dim: int | None = None,
        backend_name: str = "aer_simulator_statevector",
        noise_model: None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.input_dim = input_dim
        self.weight_dim = weight_dim or depth * num_qubits * 3  # RX,RZ,RY per layer

        # Parameter vectors
        self.input_params = ParameterVector("x", input_dim)
        self.weight_params = ParameterVector("w", self.weight_dim)

        # Build the circuit
        self.circuit = self._build_circuit()

        # Observables: sum of Pauli‑Y on each qubit
        y_ops = ["Y" * num_qubits]
        self.observables = [SparsePauliOp.from_list([(op, 1.0)]) for op in y_ops]

        # Estimator primitive
        self.estimator = StatevectorEstimator(
            backend=Aer.get_backend(backend_name),
            noise_model=noise_model,
        )

        # QNN wrapper
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=[self.input_params],
            weight_params=[self.weight_params],
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Input encoding: angle‑encoding on each qubit
        for i, param in enumerate(self.input_params):
            qc.ry(param, i)
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(self.weight_params[idx], q); idx += 1
                qc.rz(self.weight_params[idx], q); idx += 1
                qc.rx(self.weight_params[idx], q); idx += 1
            # Entanglement
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the QNN for a batch of input samples.
        `inputs` should be an array of shape (batch, input_dim).
        Returns expectation values of the observables.
        """
        return self.qnn.predict(inputs)  # uses the underlying Estimator primitive

    def set_weights(self, weights: np.ndarray) -> None:
        """Update the weight parameters of the circuit."""
        self.qnn.set_weights(weights)

    def get_weights(self) -> np.ndarray:
        """Retrieve current trainable weights."""
        return self.qnn.get_weights()


__all__ = ["EstimatorQNNGen208"]
