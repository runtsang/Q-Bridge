"""QCNNGen119: Classical hybrid network with a quantum expectation head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from.quantum_wrapper import QuantumCircuitWrapper, HybridFunction


class QCNNGen119Model(nn.Module):
    """
    Classical convolutional network that mirrors the QCNN structure and
    terminates with a differentiable hybrid head. The hybrid head
    evaluates a lightweight 2‑qubit circuit, enabling efficient
    back‑propagation of quantum gradients.
    """

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Convolution‑pool modules
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        # Dropout regularisers
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid quantum head
        self.hybrid = Hybrid(self.fc3.out_features, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

        # Hybrid expectation head
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a 2‑qubit quantum
    circuit. It uses QuantumCircuitWrapper to keep the interface
    lightweight and fully differentiable.
    """

    def __init__(self, n_qubits: int, shift: float = 0.0) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits=n_qubits, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.circuit.shift)


class HybridFunction(torch.autograd.Function):
    """
    Custom autograd function that calls the wrapped quantum circuit.
    The forward pass returns the circuit expectation value; the backward
    pass approximates the gradient via parameter‑shift.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        circuit: QuantumCircuitWrapper,
        shift: float,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = torch.ones_like(inputs) * ctx.shift

        # Parameter‑shift finite‑difference approximation
        gradients = []
        for val in inputs.tolist():
            pos = ctx.circuit.run([val + shift.item()])
            neg = ctx.circuit.run([val - shift.item()])
            gradients.append(pos - neg)
        gradients = torch.tensor(gradients).float()
        return gradients * grad_output, None, None


__all__ = ["QCNNGen119Model", "Hybrid", "HybridFunction"]
