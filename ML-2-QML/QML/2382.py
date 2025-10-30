from __future__ import annotations

import numpy as np
import qiskit
from qiskit import execute, assemble, transpile
from qiskit.circuit.random import random_circuit
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumCircuit:
    """Two‑qubit variational circuit used as a quantum expectation head."""
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
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.tolist()):
            right = ctx.circuit.run([value + shift[idx]])
            left = ctx.circuit.run([value - shift[idx]])
            gradients.append(right - left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class HybridHead(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, in_features: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(in_features, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QuanvCircuit:
    """Quantum filter that emulates a quanvolution layer."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class QuantumFilter(nn.Module):
    """Wraps QuanvCircuit to provide a PyTorch‑friendly interface."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 0.0) -> None:
        super().__init__()
        self.filter = QuanvCircuit(kernel_size, backend, shots, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, H, W)
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            channel_vals = []
            for c in range(x.shape[1]):
                channel = x[i, c, :, :].cpu().numpy()
                channel_vals.append(self.filter.run(channel))
            outputs.append(np.mean(channel_vals))
        return torch.tensor(outputs, dtype=torch.float32, device=x.device)

class HybridQCNet(nn.Module):
    """Hybrid CNN that combines a quantum filter and a quantum expectation head."""
    def __init__(self, use_filter: bool = True, filter_kwargs: dict | None = None) -> None:
        super().__init__()
        self.use_filter = use_filter
        if use_filter:
            backend_filter = qiskit.Aer.get_backend("qasm_simulator")
            self.quantum_filter = QuantumFilter(**(filter_kwargs or {
                "kernel_size": 2,
                "backend": backend_filter,
                "shots": 100,
                "threshold": 127
            }))
        else:
            self.quantum_filter = None

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        backend_head = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid_head = HybridHead(2, backend_head, shots=100, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_filter:
            # compute quantum filter output per sample
            filter_out = self.quantum_filter(x)  # shape: (batch,)
            filter_out = filter_out.unsqueeze(1)  # (batch, 1)
        else:
            filter_out = torch.zeros(x.shape[0], 1, device=x.device)

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
        x = self.fc3(x)  # (batch, 1)

        # concatenate filter output with fc3 output
        combined = torch.cat((x, filter_out), dim=1)  # (batch, 2)
        probs = self.hybrid_head(combined)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQCNet", "QuantumCircuit", "HybridHead", "QuantumFilter"]
