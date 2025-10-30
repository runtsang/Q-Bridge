from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from typing import List

class FraudDetectionHybrid:
    """
    Quantum implementation that mirrors the classical fraud detection pipeline.
    Uses a variational circuit with two qubits (one per input feature) and
    a StatevectorEstimator to produce a scalar output.
    """
    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        # Build a simple circuit: RX for inputs, followed by a variational layer
        self.circuit = QuantumCircuit(num_qubits)
        self.input_params: List[Parameter] = [Parameter(f"x{i}") for i in range(num_qubits)]
        self.weight_params: List[Parameter] = [Parameter(f"theta{i}") for i in range(num_qubits * 2)]

        # Input encoding: RX gates
        for i in range(num_qubits):
            self.circuit.rx(self.input_params[i], i)

        # Variational block: two layers of RY and RZ
        for i in range(num_qubits):
            self.circuit.ry(self.weight_params[i], i)
        for i in range(num_qubits):
            self.circuit.rz(self.weight_params[num_qubits + i], i)

        # Measurement expectation
        self.circuit.measure_all()

        # Estimator and QNN wrapper
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
            output_shape=1,
            interpret=lambda x: x[0],  # Return expectation of the first qubit
        )

    def get_qnn(self) -> EstimatorQNN:
        """
        Return the underlying EstimatorQNN instance which can be used as a
        torch.nn.Module via qiskit_machine_learning.neural_networks.EstimatorQNN.
        """
        return self.qnn

__all__ = ["FraudDetectionHybrid"]
