import numpy as np
from qiskit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

def _conv_block(params: Sequence[ParameterVector]) -> QuantumCircuit:
    sub = QuantumCircuit(2)
    sub.rz(-np.pi/2, 1)
    sub.cx(1, 0)
    sub.rz(params[0], 0)
    sub.ry(params[1], 1)
    sub.cx(0, 1)
    sub.ry(params[2], 1)
    sub.cx(1, 0)
    sub.rz(np.pi/2, 0)
    return sub

def _pool_block(params: Sequence[ParameterVector]) -> QuantumCircuit:
    sub = QuantumCircuit(2)
    sub.rz(-np.pi/2, 1)
    sub.cx(1, 0)
    sub.rz(params[0], 0)
    sub.ry(params[1], 1)
    sub.cx(0, 1)
    sub.ry(params[2], 1)
    return sub

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_block(params[i*3:(i+2)*3])
        qc.append(sub, [i, i+1])
    return qc

def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = _pool_block(params[i*3:(i+2)*3])
        qc.append(sub, [i, i+1])
    return qc

def self_attention_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="SelfAttention Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3 + num_qubits - 1)
    for i in range(num_qubits):
        qc.rx(params[3*i], i)
        qc.ry(params[3*i + 1], i)
        qc.rz(params[3*i + 2], i)
    for i in range(num_qubits - 1):
        qc.crx(params[num_qubits*3 + i], i, i+1)
    return qc

def QCNN() -> EstimatorQNN:
    estimator = Estimator()
    n_qubits = 8

    # Feature map
    feature_map = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        feature_map.ry(ParameterVector("x_" + str(i), 1)[0], i)

    # Ansatz construction
    ansatz = QuantumCircuit(n_qubits, name="Ansatz")

    ansatz.compose(conv_layer(n_qubits, "c1"), list(range(n_qubits)), inplace=True)
    ansatz.compose(pool_layer(n_qubits, "p1"), list(range(n_qubits)), inplace=True)
    ansatz.compose(conv_layer(n_qubits//2, "c2"), list(range(n_qubits//2, n_qubits)), inplace=True)
    ansatz.compose(pool_layer(n_qubits//2, "p2"), list(range(n_qubits//2, n_qubits)), inplace=True)
    ansatz.compose(conv_layer(n_qubits//4, "c3"), list(range(n_qubits//4*3, n_qubits)), inplace=True)
    ansatz.compose(pool_layer(n_qubits//4, "p3"), list(range(n_qubits//4*3, n_qubits)), inplace=True)

    # Insert self‑attention block after the final pooling layer
    ansatz.compose(self_attention_layer(n_qubits, "sa1"), list(range(n_qubits)), inplace=True)

    # Observable: single Pauli‑Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits-1), 1)])

    return EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

class QuantumHybridQuanvolutionClassifier(nn.Module):
    """Hybrid quantum‑classical classifier that uses a QCNN with a self‑attention layer
    followed by a classical linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qnn = QCNN()
        self.head = nn.Linear(1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Flatten input to match feature map size
        x = x.view(x.size(0), -1)
        q_out = self.qnn(x)
        logits = self.head(q_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumHybridQuanvolutionClassifier"]
