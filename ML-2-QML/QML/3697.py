"""Quantum implementation of QCNNFraudHybrid.

The class wraps an EstimatorQNN built from a convolution‑pooling circuit.
It exposes a forward method compatible with PyTorch for end‑to‑end training.
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# 1. Basic two‑qubit primitives
# --------------------------------------------------------------------------- #
def _conv_unitary(params: ParameterVector) -> QuantumCircuit:
    circ = QuantumCircuit(2)
    circ.rz(-np.pi / 2, 1)
    circ.cx(1, 0)
    circ.rz(params[0], 0)
    circ.ry(params[1], 1)
    circ.cx(0, 1)
    circ.ry(params[2], 1)
    circ.cx(1, 0)
    circ.rz(np.pi / 2, 0)
    return circ

def _pool_unitary(params: ParameterVector) -> QuantumCircuit:
    circ = QuantumCircuit(2)
    circ.rz(-np.pi / 2, 1)
    circ.cx(1, 0)
    circ.rz(params[0], 0)
    circ.ry(params[1], 1)
    circ.cx(0, 1)
    circ.ry(params[2], 1)
    return circ

# --------------------------------------------------------------------------- #
# 2. Layer constructors
# --------------------------------------------------------------------------- #
def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_unitary(params[i*3:(i+2)*3])
        qc.append(sub, [i, i+1])
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        sub = _pool_unitary(params[(i//2)*3:((i//2)+1)*3])
        qc.append(sub, [i, i+1])
    return qc

# --------------------------------------------------------------------------- #
# 3. Custom feature map
# --------------------------------------------------------------------------- #
def _linear_feature_map(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(ParameterVector(f'x_{i}', 1)[0], i)
    return qc

# --------------------------------------------------------------------------- #
# 4. Build the QCNN ansatz
# --------------------------------------------------------------------------- #
def QCNNQuantum() -> EstimatorQNN:
    """Builds an EstimatorQNN representing the QCNN architecture."""
    # Feature map
    fm = _linear_feature_map(8)

    # Ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), inplace=True)

    # Observable: single‑qubit Z on the last qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    return EstimatorQNN(
        circuit=fm.compose(ansatz),
        observables=observable,
        input_params=fm.parameters,
        weight_params=ansatz.parameters,
        estimator=StatevectorEstimator(),
    )

# --------------------------------------------------------------------------- #
# 5. Quantum hybrid wrapper
# --------------------------------------------------------------------------- #
class QCNNFraudHybrid(torch.nn.Module):
    """Quantum QCNN network with a simple readout."""
    def __init__(self) -> None:
        super().__init__()
        self.qnn = QCNNQuantum()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert torch tensor to numpy array
        data = inputs.detach().cpu().numpy()
        # Run the QNN
        preds = self.qnn.predict(data)
        return torch.tensor(preds, dtype=inputs.dtype, device=inputs.device)

__all__ = ["QCNNFraudHybrid"]
