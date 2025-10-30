from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np
import torch

def generate_superposition_data(num_features: int, samples: int):
    """Generate synthetic superposition data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridEstimatorQNN:
    """Quantum estimator that blends a parameterised Qiskit circuit with a classical read‑out.

    The circuit is inspired by the Quantum‑NAT architecture: a random layer,
    trainable single‑qubit rotations and a small entangling block.
    A StatevectorEstimator evaluates the expectation value of Pauli‑Z,
    which is used as the regression output.
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.input_params = [Parameter(f"x{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"w{i}") for i in range(n_qubits)]
        self.circuit = self._build_circuit()
        self.observables = SparsePauliOp.from_list([("Z" * n_qubits, 1)])
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        # Random layer: Hadamard + RZ with fixed angles
        for i in range(self.n_qubits):
            qc.h(i)
            qc.rz(Parameter(f"rnd{i}"), i)
        # Trainable rotations
        for i in range(self.n_qubits):
            qc.rx(self.weight_params[i], i)
            qc.ry(self.weight_params[i], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit on a batch of classical inputs."""
        param_dict = {f"x{i}": val for i, val in enumerate(inputs)}
        result = self.estimator.run(self.circuit, param_dict, self.observables)
        return result.values[0].real

__all__ = ["HybridEstimatorQNN", "RegressionDataset", "generate_superposition_data"]
