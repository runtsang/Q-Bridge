import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex states and regression targets."""
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
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields complex states and targets for regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QCNNRegressionHybrid:
    """Quantum QCNN-inspired regression model using EstimatorQNN."""
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        estimator = Estimator()

        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi / 2, 0)
            return qc

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for i in range(0, num_qubits, 2):
                qc.compose(conv_circuit(params[i//2*3:i//2*3+3]), [i, i + 1], inplace=True)
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=len(sources) * 3)
            for i, (src, snk) in enumerate(zip(sources, sinks)):
                qc.compose(pool_circuit(params[i*3:i*3+3]), [src, snk], inplace=True)
            return qc

        feature_map = ZFeatureMap(self.num_qubits)

        ansatz = QuantumCircuit(self.num_qubits)
        ansatz.compose(conv_layer(self.num_qubits, "c1"), inplace=True)
        ansatz.compose(pool_layer(list(range(self.num_qubits//2)), list(range(self.num_qubits//2, self.num_qubits)), "p1"), inplace=True)
        ansatz.compose(conv_layer(self.num_qubits//2, "c2"), inplace=True)
        ansatz.compose(pool_layer(list(range(self.num_qubits//4)), list(range(self.num_qubits//4, self.num_qubits//2)), "p2"), inplace=True)
        ansatz.compose(conv_layer(self.num_qubits//4, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

        self.qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.qnn(state_batch).squeeze(-1)

__all__ = ["QCNNRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
