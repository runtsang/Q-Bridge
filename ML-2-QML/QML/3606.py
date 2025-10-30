"""Quantum estimator that replaces the classical ConvFilter with a variational
convolutional circuit and uses a Qiskit EstimatorQNN for regression."""
import numpy as np
import qiskit
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class QuanvCircuit:
    """Variational convolutional circuit used as a feature extractor."""
    def __init__(self, kernel_size: int, backend, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(self.n_qubits)
        self.data_params = [Parameter(f"x{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.data_params[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, depth=2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Return average |1> probability over all qubits."""
        data = data.reshape(-1, self.n_qubits)
        param_binds = [{p: np.pi if val > self.threshold else 0.0 for p, val in zip(self.data_params, d)}
                       for d in data]
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        prob = 0.0
        for bitstring, c in counts.items():
            ones = bitstring.count('1')
            prob += ones * c
        return prob / (self.shots * self.n_qubits)

class EstimatorQNN:
    """Quantum neural network that embeds 2‑D data with a variational filter
    and performs regression with a trainable quantum circuit."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5,
                 obs: SparsePauliOp | None = None):
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.filter = QuanvCircuit(kernel_size, backend, threshold=threshold)
        self.circuit = QuantumCircuit(self.filter.n_qubits)
        self.weight_params = [Parameter(f"w{i}") for i in range(1, self.filter.n_qubits + 1)]
        for i, w in enumerate(self.weight_params, start=1):
            self.circuit.ry(w, i-1)
        if obs is None:
            obs = SparsePauliOp.from_list([("Z" * self.filter.n_qubits, 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=obs,
            input_params=[],
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(self, data: np.ndarray) -> float:
        """Compute regression output for 2‑D input."""
        feature = self.filter.run(data)
        param_binds = {self.weight_params[0]: 2 * np.arcsin(feature)}
        result = self.estimator_qnn.run(param_binds)
        return result[0].real

__all__ = ["EstimatorQNN"]
