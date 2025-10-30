import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params: np.ndarray) -> QuantumCircuit:
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

def conv_layer(num_qubits: int, name: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(name, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i * 3:(i + 2) * 3]), [i, i + 1])
    return qc

def pool_circuit(params: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, name: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(name, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(pool_circuit(params[:3]), [src, snk])
        params = params[3:]
    return qc

def build_qcnn_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    feature_map = ZFeatureMap(num_qubits)
    qc.append(feature_map, range(num_qubits))

    # First convolution + pooling layer
    qc.append(conv_layer(num_qubits, "c1"), range(num_qubits))
    qc.append(pool_layer(list(range(num_qubits//2)), list(range(num_qubits//2, num_qubits)), "p1"),
               range(num_qubits))

    # Second convolution + pooling layer
    qc.append(conv_layer(num_qubits//2, "c2"),
               range(num_qubits//2, num_qubits))
    qc.append(pool_layer([0, 1], [2, 3], "p2"), range(num_qubits//2, num_qubits))

    # Third convolution + pooling layer
    qc.append(conv_layer(num_qubits//4, "c3"),
               range(num_qubits//4, num_qubits))
    qc.append(pool_layer([0], [1], "p3"), range(num_qubits//4, num_qubits))

    return qc

def QCNNHybridQNN(num_qubits: int) -> EstimatorQNN:
    """
    Returns a variational quantum neural network that implements the QCNN
    architecture.  The returned object can be used as a layer in a PyTorch model.
    """
    estimator = Estimator()
    circuit = build_qcnn_circuit(num_qubits)
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    # Parameters that encode input features
    input_params = [p for p in circuit.parameters if "feature" in p.name]
    # Remaining parameters are trainable weights
    weight_params = [p for p in circuit.parameters if "c" in p.name or "p" in p.name]

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return qnn

# ------------------------------
# Regression data helpers (adapted from QuantumRegression seed)
# ------------------------------

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["QCNNHybridQNN", "RegressionDataset", "generate_superposition_data"]
