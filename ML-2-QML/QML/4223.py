from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """
    Two‑qubit variational circuit used as a quantum head.
    Parameters are learned by the surrounding PyTorch module.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1000) -> None:
        self.backend = AerSimulator()
        self.n_qubits = n_qubits
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_q = list(range(n_qubits))
        self.circuit.h(all_q)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_q)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        def expectation(count_dict):
            probs = np.array(list(count_dict.values())) / self.shots
            states = np.array([int(k, 2) for k in count_dict.keys()])
            return np.sum(states * probs)
        return np.array([expectation(counts)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit.
    Uses finite‑difference to approximate gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor,
                circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectation = circuit.run(thetas).squeeze()
        result = torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            e_plus = ctx.circuit.run(np.array([val + shift])).squeeze()
            e_minus = ctx.circuit.run(np.array([val - shift])).squeeze()
            grads.append(e_plus - e_minus)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None

class FraudHybridQuantumModel(nn.Module):
    """
    Hybrid model that mirrors the classical backbone but replaces the
    photonic layer with a parameterised two‑qubit quantum head.
    Supports both binary classification and regression.
    """
    def __init__(self,
                 regression: bool = False,
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        self.regression = regression
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.quantum_circuit = QuantumCircuitWrapper()
        self.quantum_head = nn.Linear(self.fc3.out_features, 1)
        self.shift = np.pi / 2

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

        if self.regression:
            return self.quantum_head(x).squeeze(-1)
        else:
            logits = self.quantum_head(x).squeeze(-1)
            probs = torch.sigmoid(logits)
            return torch.cat([probs, 1 - probs], dim=-1)

    def get_quantum_circuit(self) -> qiskit.QuantumCircuit:
        """Return the underlying Qiskit circuit for inspection."""
        return self.quantum_circuit.circuit

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "FraudHybridQuantumModel"]
