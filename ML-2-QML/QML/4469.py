"""Unified quantum‑hybrid model.

The module implements a feed‑forward network that feeds into a variational quantum circuit
for classification or regression.  The same class structure is reused for a pure
classical baseline, and a quantum‑kernel wrapper is provided for kernel‑based methods.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit import Aer

# --------------------------------------------------------------------------- #
#  Classical backbone – dense feed‑forward network (shared with the ML side)
# --------------------------------------------------------------------------- #
class _DenseBackbone(nn.Module):
    """Same as in the classical module."""
    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None, activation: nn.Module = nn.ReLU()):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation)
            input_dim = h
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Wrapper around a parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
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
        """Execute the parametrised circuit for the provided angles."""
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
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
#  Hybrid head (quantum expectation)
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit

        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Hybrid CNN (quantum head)
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

# --------------------------------------------------------------------------- #
#  Quantum kernel
# --------------------------------------------------------------------------- #
class QuantumKernel:
    """Quantum kernel evaluated via a fixed Qiskit ansatz."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.simulator = AerSimulator()

    def _encode(self, circuit: qiskit.QuantumCircuit, params: np.ndarray):
        for i, param in enumerate(params):
            circuit.ry(param, i)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        circuit_x = qiskit.QuantumCircuit(self.n_qubits)
        self._encode(circuit_x, x)
        circuit_y = qiskit.QuantumCircuit(self.n_qubits)
        self._encode(circuit_y, y)

        result_x = self.simulator.run(circuit_x).result()
        result_y = self.simulator.run(circuit_y).result()
        state_x = result_x.get_statevector(circuit_x)
        state_y = result_y.get_statevector(circuit_y)
        overlap = np.vdot(state_x, state_y)
        return abs(overlap) ** 2

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel(n_qubits=4, backend=AerSimulator(), shots=1024)
    return np.array([[kernel.evaluate(x.numpy(), y.numpy()) for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Quantum regression dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Unified model (quantum variant)
# --------------------------------------------------------------------------- #
class UnifiedQuantumHybridModel(nn.Module):
    """
    Unified quantum model that can act as a hybrid head or a full quantum classifier.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_sizes : list[int] | None
        Hidden layer sizes for the dense backbone.
    task : str
        Either 'classification' or'regression'.
    use_cnn : bool
        If True, the model uses the CNN+Hybrid head defined in QCNet.
    """
    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None,
                 task: str = "classification", use_cnn: bool = False):
        super().__init__()
        self.task = task
        self.use_cnn = use_cnn
        if use_cnn:
            self.base = QCNet()
        else:
            self.backbone = _DenseBackbone(input_dim, hidden_sizes)
            last_dim = self.backbone.layers[-1].out_features
            backend = Aer.get_backend("aer_simulator")
            if task == "classification":
                self.head = Hybrid(last_dim, backend, shots=100, shift=np.pi / 2)
            else:
                self.head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            return self.base(x)
        features = self.backbone(x)
        out = self.head(features)
        if self.task == "classification":
            prob = out
            return torch.cat((prob, 1 - prob), dim=-1)
        return out

__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "Hybrid",
    "QCNet",
    "QuantumKernel",
    "kernel_matrix",
    "generate_superposition_data",
    "RegressionDataset",
    "UnifiedQuantumHybridModel",
]
