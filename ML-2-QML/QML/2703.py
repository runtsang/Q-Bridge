import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index+3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

class HybridQCNNBinaryClassifier(nn.Module):
    """
    Quantum‑inspired QCNN implemented with an EstimatorQNN.
    The circuit mirrors the classical feature‑map and convolutional
    layers defined in the reference, enabling differentiable training
    via PyTorch autograd.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # Build ansatz
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4,8)), inplace=True)
        ansatz.compose(pool_layer([0,1], [2,3], "p2"), list(range(4,8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6,8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6,8)), inplace=True)
        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
        # Estimator
        estimator = Estimator()
        # QNN
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 8) float tensor
        x_np = x.detach().cpu().numpy()
        preds = self.qnn.forward(x_np)  # expectation values, shape (batch,)
        preds = torch.tensor(preds, dtype=torch.float32, device=x.device)
        probs = torch.sigmoid(preds)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQCNNBinaryClassifier"]
