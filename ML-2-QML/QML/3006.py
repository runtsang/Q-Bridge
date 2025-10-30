from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
import numpy as np
import torch

def generate_superposition_data(num_wires: int, samples: int):
    """
    Generate quantum states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>
    and corresponding regression targets.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class QuantumRegressionDataset(torch.utils.data.Dataset):
    """
    Quantum dataset providing state vectors and regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionQNN:
    """
    Wrapper around Qiskit EstimatorQNN that evaluates a parameterized quantum circuit
    for regression. The input parameter 'theta' is supplied by the user; the weight
    parameter 'phi' is trainable by the EstimatorQNN optimizer.
    """
    def __init__(self, num_qubits: int = 1):
        self.num_qubits = num_qubits
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.ry(theta, 0)
        qc.rx(phi, 0)
        observable = SparsePauliOp.from_list([("Y" * num_qubits, 1)])
        estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[theta],
            weight_params=[phi],
            estimator=estimator
        )

    def forward(self, theta_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum neural network on a batch of input angles.
        """
        inputs = {"theta": theta_batch}
        preds = self.qnn.forward(inputs)
        return np.array(preds)

__all__ = ["QuantumRegressionQNN", "QuantumRegressionDataset", "generate_superposition_data"]
