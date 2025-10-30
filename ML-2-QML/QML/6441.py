"""Hybrid quantum binary classifier with a variational expectation head.

The quantum head is implemented with a two‑qubit parameterised circuit executed on
Qiskit’s Aer simulator.  The circuit uses the parameter‑shift rule for
gradient estimation, making it fully differentiable inside PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import qiskit
from qiskit import assemble, transpile


# --------------------------------------------------------------------------- #
# Dataset utilities (identical to the classical version for consistency)
# --------------------------------------------------------------------------- #
def generate_superposition_data_binary(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = (np.sin(angles) + 0.1 * np.cos(2 * angles)) > 0
    return x, y.astype(np.float32)


class BinaryClassificationDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data_binary(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """Two‑qubit variational circuit with a single rotation parameter."""

    def __init__(self, backend, shots: int = 1000):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("θ")
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def run(self, theta: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of theta values and return the Z‑expectation."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in theta],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()

        def expectation(count_dict):
            probs = np.array(list(count_dict.values())) / self.shots
            # Map bitstring to integer (00->0, 01->1, 10->2, 11->3)
            states = np.array([int(k, 2) for k in count_dict.keys()])
            return np.sum(states * probs)

        if isinstance(counts, list):
            return np.array([expectation(c) for c in counts])
        return np.array([expectation(counts)])

# --------------------------------------------------------------------------- #
# Hybrid quantum head
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards values through the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to numpy for the simulator
        theta_vals = inputs.squeeze().cpu().numpy()
        exp_vals = ctx.circuit.run(theta_vals)
        result = torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = np.ones_like(inputs.squeeze().cpu().numpy()) * ctx.shift
        grad_vals = []
        for val, s in zip(inputs.squeeze().cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])[0]
            left = ctx.circuit.run([val - s])[0]
            grad_vals.append(right - left)
        grad = torch.tensor(grad_vals, device=inputs.device, dtype=torch.float32)
        return grad * grad_output, None, None


class HybridQuantumHead(nn.Module):
    """Quantum expectation head that uses the parameter‑shift rule."""

    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2, shots: int = 500):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(qiskit.Aer.get_backend("aer_simulator"), shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
# Unified classifier
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    CNN followed by a hybrid quantum head.  The head is a variational
    expectation layer that can be swapped out for a classical sigmoid if
    desired, but the quantum implementation is fully differentiable.
    """

    def __init__(self, use_quantum_head: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if use_quantum_head:
            self.hybrid = HybridQuantumHead()
        else:
            self.hybrid = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier", "HybridQuantumHead", "QuantumCircuitWrapper", "BinaryClassificationDataset"]
