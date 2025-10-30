"""Quantum components for AdvancedQCNN."""

from __future__ import annotations

import numpy as np
from typing import Sequence

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

import torch
import torchquantum as tq

# 1. QCNN circuit builder
def build_qcnn_circuit(num_qubits: int = 8) -> QuantumCircuit:
    """Constructs the QCNN ansatz as described in the reference."""
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
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
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
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc.compose(pool_circuit(params[param_index:param_index + 3]), [source, sink], inplace=True)
            qc.barrier()
            param_index += 3
        return qc

    # Build ansatz
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(conv_layer(num_qubits // 2, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(num_qubits // 4, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Feature map
    feature_map = ZFeatureMap(num_qubits)

    # Full circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    return circuit

# 2. Quantum self‑attention circuit
class QuantumSelfAttention:
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

# 3. Quantum kernel using TorchQuantum
class QuantumKernel:
    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])
        for wire in range(self.n_wires):
            self.q_device.ry(x[:, wire], wire)
        for wire in range(self.n_wires):
            self.q_device.ry(-y[:, wire], wire)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# 4. Quantum counterpart of AdvancedQCNN
class AdvancedQCNNQML:
    """Quantum counterpart of AdvancedQCNN. Provides QCNN circuit, self‑attention, and quantum kernel."""
    def __init__(self, num_qubits: int = 8, n_qubits_attention: int = 4, n_wires_kernel: int = 4):
        self.num_qubits = num_qubits
        self.cnn_circuit = build_qcnn_circuit(num_qubits)
        self.attention = QuantumSelfAttention(n_qubits_attention)
        self.kernel = QuantumKernel(n_wires_kernel)

    def get_qcnn_circuit(self) -> QuantumCircuit:
        return self.cnn_circuit

    def run_attention(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        return self.attention.run(rotation_params, entangle_params, shots)

    def compute_kernel(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

__all__ = [
    "build_qcnn_circuit",
    "QuantumSelfAttention",
    "QuantumKernel",
    "kernel_matrix",
    "AdvancedQCNNQML",
]
