"""HybridQCNN: quantum implementation of the hybrid convolutional network.

The module builds a QCNN ansatz consisting of convolutional and pooling
layers defined in the original QCNN.py reference.  It then wraps the
Ansatz in an EstimatorQNN and exposes a PyTorch‑style forward method.
The final classification head is a single linear layer that maps the
quantum expectation to a binary probability.

The class can be used as a drop‑in replacement for the classical
HybridQCNN, but relies on a quantum backend (Aer simulator or a real
device) for the convolutional stack.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# 1. Quantum convolution and pooling primitives
# --------------------------------------------------------------------------- #
def conv_circuit(params):
    """Two‑qubit convolution block used in QCNN."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# --------------------------------------------------------------------------- #
# 2. HybridQCNN: quantum backbone + hybrid head
# --------------------------------------------------------------------------- #
class HybridQCNN(nn.Module):
    """Quantum convolutional network with a hybrid classification head."""
    def __init__(self, backend=None, shots=1024):
        super().__init__()
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.estimator = Estimator()

        # Feature map
        self.feature_map = ZFeatureMap(8)

        # Build ansatz with convolution and pooling layers
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Full circuit
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        # Observable for expectation value
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

        # Final classification head
        self.classifier = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs shape [batch, 8]
        quantum_out = self.qnn(inputs)          # shape [batch, 1]
        logits = self.classifier(quantum_out)   # shape [batch, 1]
        probs = torch.sigmoid(logits)
        return probs

__all__ = ["HybridQCNN"]
