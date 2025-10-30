from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridEstimatorQNN:
    """
    Quantum neural network that mirrors the hybrid classical architecture:
    - A Z‑feature map encodes classical data.
    - A convolution‑style ansatz entangles qubits.
    - Expectation value of a Pauli‑Z observable is returned as the prediction.
    """
    def __init__(self, num_qubits: int = 8, depth: int = 2):
        # Feature map
        self.feature_map = ZFeatureMap(num_qubits, reps=1)
        # Ansatz: repeated conv‑pool layers with parameters
        self.ansatz = self._build_ansatz(num_qubits, depth)

        # Observable: target qubit measurement
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator
        )

    def _build_ansatz(self, num_qubits: int, depth: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_prefix = "θ"
        params = ParameterVector(param_prefix, length=num_qubits * depth)
        idx = 0
        for d in range(depth):
            # Rotate each qubit with a trainable parameter
            for i in range(num_qubits):
                qc.rx(params[idx], i)
                idx += 1
            # Entangle neighboring qubits
            for i in range(0, num_qubits - 1, 2):
                qc.cx(i, i + 1)
        qc.barrier()
        return qc

    def predict(self, data: list[float]) -> float:
        """
        Evaluate the QNN on a single data point.
        """
        if len(data)!= len(self.feature_map.parameters):
            raise ValueError(f"Expected {len(self.feature_map.parameters)} input values, got {len(data)}")
        param_dict = {str(p): val for p, val in zip(self.feature_map.parameters, data)}
        result = self.estimator_qnn.predict(param_dict)
        # Expectation value is a single float
        return float(result[0])

__all__ = ["HybridEstimatorQNN"]
