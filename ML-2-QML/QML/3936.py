"""Quantum kernel and QCNN implementation.

This module provides the quantum counterparts of the classical
components defined in ``QuantumKernelMethod__gen252.py``.  It relies
on TorchQuantum for the kernel and on Qiskit for the convolution‑pooling
ansatz.  All objects are lightweight and designed to be imported by
the hybrid façade without side effects.
"""

import numpy as np
import torch
from typing import Sequence
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Quantum kernel -----------------------------------------------------------

class QuantumRBFKernel(tq.QuantumModule):
    """Quantum RBF kernel based on a fixed list of Ry gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Convenient wrapper that exposes a four‑wire kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumRBFKernel(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Quantum QCNN -------------------------------------------------------------

import json
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Helper circuits ----------------------------------------------------------

def _conv_circuit(params):
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

def _pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    return QuantumCircuit(num_qubits).append(qc_inst, qubits)

def _pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(_pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    return QuantumCircuit(num_qubits).append(qc_inst, range(num_qubits))

def QCNNQuantumCircuit() -> EstimatorQNN:
    """Builds the QCNN ansatz and returns an EstimatorQNN instance."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolution and pooling
    ansatz.compose(_conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second convolution and pooling
    ansatz.compose(_conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third convolution and pooling
    ansatz.compose(_conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# Hybrid QML façade --------------------------------------------------------

class QuantumHybridKernelQCNNQML:
    """
    Quantum‑only counterpart of :class:`HybridKernelQCNN` defined in the
    classical module.  It exposes the quantum kernel matrix and the QCNN
    forward pass via EstimatorQNN.
    """
    def __init__(self, gamma: float = 1.0, n_wires: int = 4) -> None:
        self.gamma = gamma
        self.n_wires = n_wires
        self.kernel = QuantumKernel()
        self.qcnn = QCNNQuantumCircuit()

    def kernel_matrix(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        return quantum_kernel_matrix(X, X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # EstimatorQNN expects a NumPy array; convert and reshape
        preds = self.qcnn.predict(X.detach().numpy())
        return torch.tensor(preds).reshape(-1, 1)

__all__ = [
    "QuantumRBFKernel",
    "QuantumKernel",
    "quantum_kernel_matrix",
    "QCNNQuantumCircuit",
    "QuantumHybridKernelQCNNQML",
]
