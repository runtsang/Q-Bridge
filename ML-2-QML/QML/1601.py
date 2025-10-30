"""QCNNGen167 – a modular, parameter‑shiftable quantum QCNN implementation."""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import Adam
from qiskit_machine_learning.utils import algorithm_globals
from typing import Iterable, Tuple


def _conv_circuit(params: ParameterVector, qubits: Tuple[int, int]) -> QuantumCircuit:
    """Two‑qubit convolution block using parameter‑shiftable RZ/RY gates."""
    qc = QuantumCircuit(*qubits)
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(np.pi / 2, qubits[0])
    return qc


def _pool_circuit(params: ParameterVector, qubits: Tuple[int, int]) -> QuantumCircuit:
    """Two‑qubit pooling block – analogous to a measurement‑based reduction."""
    qc = QuantumCircuit(*qubits)
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    return qc


def _layer(num_pairs: int, prefix: str, layer_type: str) -> QuantumCircuit:
    """Construct a convolution or pooling layer over paired qubits."""
    qc = QuantumCircuit(num_pairs * 2)
    params = ParameterVector(prefix, length=num_pairs * 3)
    for i in range(num_pairs):
        q1, q2 = 2 * i, 2 * i + 1
        sub = (
            _conv_circuit if layer_type == "conv" else _pool_circuit
        )(params[3 * i : 3 * i + 3], (q1, q2))
        qc.append(sub.to_instruction(), [q1, q2])
    return qc


def QCNN() -> EstimatorQNN:
    """Builds a QCNN with a 8‑qubit feature map and a variational ansatz."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map – 8‑qubit Z‑feature map
    fmap = ZFeatureMap(8, reps=1, entanglement="full")

    # Ansatz – two convolution‑pool‑convolution‑pool‑convolution‑pool
    ansatz = QuantumCircuit(8)
    ansatz.compose(_layer(4, "c1", "conv"), range(8), inplace=True)
    ansatz.compose(_layer(4, "p1", "pool"), range(8), inplace=True)
    ansatz.compose(_layer(2, "c2", "conv"), range(4, 8), inplace=True)
    ansatz.compose(_layer(2, "p2", "pool"), range(4, 8), inplace=True)
    ansatz.compose(_layer(1, "c3", "conv"), range(6, 8), inplace=True)
    ansatz.compose(_layer(1, "p3", "pool"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(fmap, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=fmap.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


def train(
    qnn: EstimatorQNN,
    data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 20,
    lr: float = 0.01,
    batch_size: int = 16,
) -> Iterable[float]:
    """Yield training loss per epoch using the Adam optimizer."""
    optimizer = Adam(qnn.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data[0], data[1]),
        batch_size=batch_size,
        shuffle=True,
    )
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(qnn.device), y.to(qnn.device).float()
            optimizer.zero_grad()
            output = qnn(x).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        yield epoch_loss / len(loader)


__all__ = ["QCNN", "train"]
