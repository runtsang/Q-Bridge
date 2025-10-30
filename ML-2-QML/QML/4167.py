"""Quantum definitions for the hybrid QCNN."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolutional block used in the QCNN ansatz."""
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
    """Build a convolutional layer that applies ``conv_circuit`` to adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i * 3 : (i + 2) * 3]), [i, i + 1])
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Apply a pooling block to each source–sink pair."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for idx, (src, snk) in enumerate(zip(sources, sinks)):
        qc.append(pool_circuit(params[idx * 3 : (idx + 1) * 3]), [src, snk])
    return qc

def build_qcnn_qnn(num_qubits: int = 8) -> EstimatorQNN:
    """Construct the full QCNN EstimatorQNN with feature map and variational ansatz."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # First convolution + pooling
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits // 2)),
                              list(range(num_qubits // 2, num_qubits)),
                              "p1"),
                   range(num_qubits), inplace=True)

    # Second convolution + pooling on reduced register
    reduced = num_qubits // 2
    ansatz.compose(conv_layer(reduced, "c2"), list(range(reduced, num_qubits)), inplace=True)
    ansatz.compose(pool_layer(list(range(reduced)),
                              list(range(reduced, reduced * 2)),
                              "p2"),
                   list(range(reduced, num_qubits)), inplace=True)

    # Third convolution + pooling on the final two qubits
    ansatz.compose(conv_layer(2, "c3"), list(range(num_qubits - 2, num_qubits)), inplace=True)
    ansatz.compose(pool_layer([num_qubits - 2], [num_qubits - 1], "p3"),
                   list(range(num_qubits - 2, num_qubits)), inplace=True)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """Create a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

def build_fcl_circuit(num_qubits: int, shots: int) -> QuantumCircuit:
    """Simple parameterized circuit for a fully‑connected layer."""
    qc = QuantumCircuit(num_qubits)
    theta = ParameterVector("theta", num_qubits)
    qc.h(range(num_qubits))
    qc.barrier()
    for qubit, t in zip(range(num_qubits), theta):
        qc.ry(t, qubit)
    qc.measure_all()
    return qc

class QNNWrapper(nn.Module):
    """Thin wrapper exposing an EstimatorQNN as a torch.Module."""

    def __init__(self, qnn: EstimatorQNN) -> None:
        super().__init__()
        self.qnn = qnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EstimatorQNN expects a 2‑D array of input features
        return self.qnn(x)

__all__ = [
    "conv_circuit",
    "conv_layer",
    "pool_circuit",
    "pool_layer",
    "build_qcnn_qnn",
    "build_classifier_circuit",
    "build_fcl_circuit",
    "QNNWrapper",
]
