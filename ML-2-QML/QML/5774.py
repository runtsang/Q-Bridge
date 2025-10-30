"""Quantum‑augmented hybrid classifier.

This module implements a variational circuit head that replaces the
classical linear layer.  It uses the parameter‑shift rule to compute
exact gradients, enabling end‑to‑end optimisation with PyTorch."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import Aer
from qiskit.quantum_info import SparsePauliOp


class QuantumCircuitWrapper:
    """
    Builds a parameterised ansatz with explicit data encoding and a
    depth‑controlled entangling layer.  The circuit is executed on Aer.
    """
    def __init__(self, n_qubits: int, depth: int, shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")

        # Parameters for data encoding and variational layers
        self.encoding = [f"x_{i}" for i in range(n_qubits)]
        self.weights = [f"theta_{i}" for i in range(n_qubits * depth)]
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Data encoding
        for i, qubit in enumerate(range(self.n_qubits)):
            qc.rx(f"x_{i}", qubit)
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                qc.ry(f"theta_{idx}", qubit)
                idx += 1
            for qubit in range(self.n_qubits - 1):
                qc.cz(qubit, qubit + 1)
        qc.measure_all()
        return qc

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter vectors.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{name: val for name, val in zip(self.circuit.parameters, vec)}
                             for vec in params]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation value of Z on the first qubit
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class QuantumHybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit.
    Implements the parameter‑shift rule for exact gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Run circuit for each input vector
        expectations = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for idx in range(inputs.shape[0]):
            vec = inputs[idx].detach().cpu().numpy()
            vec_plus = vec + shift
            vec_minus = vec - shift
            exp_plus = circuit.run([vec_plus])[0]
            exp_minus = circuit.run([vec_minus])[0]
            grads.append((exp_plus - exp_minus) / 2.0)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None


class QuantumHybridLayer(nn.Module):
    """
    Hybrid head that forwards activations through a variational circuit.
    """
    def __init__(self, in_features: int, n_qubits: int = 2, depth: int = 2,
                 shift: float = np.pi / 2, shots: int = 1000) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, depth, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return QuantumHybridFunction.apply(x, self.circuit, self.shift)


class HybridClassifier(nn.Module):
    """
    The full hybrid model.  The convolutional backbone remains identical
    to the classical version; only the head is quantum.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = QuantumHybridLayer(self.fc3.out_features, **kwargs)

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
        probs = self.hybrid(x).t()
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumCircuitWrapper", "QuantumHybridFunction",
           "QuantumHybridLayer", "HybridClassifier"]
