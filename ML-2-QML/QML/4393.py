"""Hybrid quantum‑classical binary classifier that fuses quantum convolution, kernel, and fraud‑detection inspired layers.

The architecture mirrors the classical version but replaces auxiliary modules with quantum counterparts:
* a 2×2 quantum convolution filter implemented with a parameterised circuit
* a quantum kernel based on a simple Ry‑Ry overlap measurement
* a fraud‑detection style quantum module that encodes parameters into rotations and entanglement

The forward pass returns a two‑class probability distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble, execute
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.random import random_circuit


class QuantumCircuit(nn.Module):
    """Parametrised two‑qubit circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
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

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)


class QuantumConvFilter(nn.Module):
    """Quantum 2×2 convolution filter implemented with a parameterised circuit."""

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch = x[..., :self.kernel_size, :self.kernel_size]
        patch = patch.mean(dim=1, keepdim=True)
        patch_np = patch.squeeze(0).cpu().numpy()
        return torch.tensor(self.run(patch_np), dtype=torch.float32).unsqueeze(-1)


class QuantumKernel(nn.Module):
    """Simple quantum kernel based on Ry‑Ry overlap measurement."""

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

    def _kernel_single(self, x: np.ndarray, y: np.ndarray) -> float:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(x[i], i)
            qc.ry(y[i], i)
        qc.measure_all()
        compiled = transpile(qc, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for key, val in counts.items():
            prob = val / self.shots
            z = 1
            for bit in key[::-1]:
                z *= (1 if bit == '0' else -1)
            exp += z * prob
        return exp

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if y is None:
            y = torch.zeros_like(x)
        batch = x.shape[0]
        out = []
        for i in range(batch):
            out.append(self._kernel_single(x[i].cpu().numpy(), y[i].cpu().numpy()))
        return torch.tensor(out, dtype=torch.float32).unsqueeze(-1)


class FraudDetectionQuantumModule(nn.Module):
    """Quantum module that emulates the photonic fraud‑detection circuit with rotations and entanglement."""

    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta1 = qiskit.circuit.Parameter("theta1")
        self.theta2 = qiskit.circuit.Parameter("theta2")
        self.circuit.ry(self.theta1, 0)
        self.circuit.ry(self.theta2, 1)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> float:
        bind = {self.theta1: params[0], self.theta2: params[1]}
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts(self.circuit)
        probs = np.array(list(result.values())) / self.shots
        exp = 0.0
        for key, prob in zip(result.keys(), probs):
            bit = key[::-1][0]
            z = 1 if bit == '0' else -1
            exp += z * prob
        return exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        out = []
        for i in range(batch):
            params = x[i].cpu().numpy()
            out.append(self.run(params))
        return torch.tensor(out, dtype=torch.float32).unsqueeze(-1)


class HybridBinaryClassifier(nn.Module):
    """Hybrid quantum‑classical binary classifier that mirrors the classical version."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)

        # Quantum auxiliary modules
        self.conv_filter = QuantumConvFilter()
        self.kernel = QuantumKernel()
        self.fraud_module = FraudDetectionQuantumModule()

        # Final linear head
        self.final_linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        fc2_out = F.relu(self.fc2(x))  # shape (batch, 84)

        # Quantum conv filter on raw input
        conv_out = self.conv_filter(inputs)  # shape (batch, 1)

        # Quantum kernel similarity (compare to zero vector)
        kernel_out = self.kernel(fc2_out)  # shape (batch, 1)

        # Combine auxiliary features
        aux = torch.cat((conv_out, kernel_out), dim=-1)  # shape (batch, 2)

        # Fraud‑detection style quantum processing
        fraud_out = self.fraud_module(aux)  # shape (batch, 1)

        logits = self.final_linear(fraud_out)  # shape (batch, 1)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier"]
