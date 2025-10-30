"""Hybrid binary classifier – quantum implementation.

The quantum module reproduces the same feature‑extraction pipeline
using a QCNN‑style circuit built with Qiskit, and offers a
differentiable hybrid head that can be attached to a classical CNN.
It also supplies a ready‑to‑use :func:`QCNN` factory that returns a
``EstimatorQNN`` for end‑to‑end training with the Qiskit Machine‑Learning
library.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit executed on a simulator."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(nn.Module):
    """Differentiable hybrid head that forwards through a quantum circuit."""

    def __init__(self, circuit: QuantumCircuitWrapper, shift: float) -> None:
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is expected to be a 1‑D tensor of angles
        expectation = self.circuit.run(x.detach().cpu().numpy())
        return torch.tensor(expectation, dtype=torch.float32, device=x.device)

    def backward(self, grad_output: torch.Tensor):
        # Shift‑rule gradient (finite‑difference)
        x = self.circuit.inputs
        shift_vec = np.ones_like(x) * self.shift
        grad = []
        for val in x:
            grad.append(
                self.circuit.run([val + self.shift]) - self.circuit.run([val - self.shift])
            )
        grad = np.array(grad, dtype=np.float32)
        return torch.tensor(grad, device=x.device) * grad_output


class HybridLayer(nn.Module):
    """Wraps the quantum head for use in a hybrid network."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift
        self.head = HybridFunction(self.quantum, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def conv_layer(num_qubits: int, param_prefix: str) -> qiskit.QuantumCircuit:
    """Build a single QCNN convolutional layer (two‑qubit blocks)."""
    qc = qiskit.QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        block = _conv_block(params[idx : idx + 3], q1, q2)
        qc.append(block, [q1, q2])
        qc.barrier()
        idx += 3
    # wrap into an instruction for reuse
    inst = qc.to_instruction()
    return qiskit.QuantumCircuit(num_qubits).append(inst, qubits)


def _conv_block(params: ParameterVector, q1: int, q2: int) -> qiskit.QuantumCircuit:
    """Internal two‑qubit convolution block used by :func:`conv_layer`."""
    block = qiskit.QuantumCircuit(2)
    block.rz(-np.pi / 2, 1)
    block.cx(1, 0)
    block.rz(params[0], 0)
    block.ry(params[1], 1)
    block.cx(0, 1)
    block.ry(params[2], 1)
    block.cx(1, 0)
    block.rz(np.pi / 2, 0)
    return block


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> qiskit.QuantumCircuit:
    """Build a pooling layer that acts on pairs of qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = qiskit.QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    idx = 0
    for src, snk in zip(sources, sinks):
        block = _pool_block(params[idx : idx + 3], src, snk)
        qc.append(block, [src, snk])
        qc.barrier()
        idx += 3
    inst = qc.to_instruction()
    return qiskit.QuantumCircuit(num_qubits).append(inst, list(range(num_qubits)))


def _pool_block(params: ParameterVector, q1: int, q2: int) -> qiskit.QuantumCircuit:
    """Internal two‑qubit pooling block."""
    block = qiskit.QuantumCircuit(2)
    block.rz(-np.pi / 2, 1)
    block.cx(1, 0)
    block.rz(params[0], 0)
    block.ry(params[1], 1)
    block.cx(0, 1)
    block.ry(params[2], 1)
    return block


def QCNN() -> EstimatorQNN:
    """Return a full QCNN QNN ready for training with Qiskit ML."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Ansatz construction – three conv/pool stages
    ansatz = qiskit.QuantumCircuit(8, name="Ansatz")

    # 1st conv
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    # 1st pool
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)

    # 2nd conv
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    # 2nd pool
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)

    # 3rd conv
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    # 3rd pool
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    circuit = qiskit.QuantumCircuit(8)
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


class HybridBinaryClassifierQML(nn.Module):
    """Hybrid CNN + QCNN head for binary classification."""

    def __init__(self, use_quantum: bool = True, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        backend = qiskit.Aer.get_backend("aer_simulator")
        self.qhead = HybridLayer(
            n_qubits=self.fc3.out_features,
            backend=backend,
            shots=200,
            shift=shift,
        ) if use_quantum else HybridFunction(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = self.qhead(x)
        return torch.cat((logits, 1 - logits), dim=-1)


__all__ = [
    "QuantumCircuitWrapper",
    "HybridFunction",
    "HybridLayer",
    "conv_layer",
    "pool_layer",
    "QCNN",
    "HybridBinaryClassifierQML",
]
