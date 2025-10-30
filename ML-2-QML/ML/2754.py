"""Hybrid classical‑quantum binary classifier – classical side.

The module implements a CNN backbone followed by a hybrid head that
processes the penultimate activations through a quantum expectation
layer.  The quantum circuit is supplied from the QML module via the
``prepare_quantum`` method.  The code focuses on the model structure
and gradient flow; training loops are left to the user.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical network that mirrors the quantum ansatz
# --------------------------------------------------------------------------- #
def build_classical_classifier(num_features: int, depth: int) -> tuple[nn.Module, list[int]]:
    """
    Construct a feed‑forward network that has the same number of
    trainable parameters as the quantum ansatz returned by
    ``build_classifier_circuit``.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 1)  # single output to encode into 1‑qubit circuit
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    return network, weight_sizes

# --------------------------------------------------------------------------- #
# Differentiable bridge between PyTorch and a quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """
    Forward pass evaluates the quantum circuit for each element of the
    input tensor.  The backward pass uses the parameter‑shift rule to
    compute gradients with respect to the classical activations.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.raw_inputs = inputs.detach().cpu().numpy().tolist()
        expect = circuit.run(ctx.raw_inputs)
        return torch.tensor(expect, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        raw_inputs = ctx.raw_inputs
        grads = []

        for val in raw_inputs:
            right = circuit.run([val + shift])[0]
            left = circuit.run([val - shift])[0]
            grads.append(right - left)

        grad_inputs = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device) * grad_output
        return grad_inputs, None, None

# --------------------------------------------------------------------------- #
# Hybrid head that combines the classical encoder with the quantum layer
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """
    Wraps a classical feed‑forward network and a quantum circuit.
    """
    def __init__(self, num_features: int, depth: int, shift: float = 0.0) -> None:
        super().__init__()
        self.classical, self.weight_sizes = build_classical_classifier(num_features, depth)
        self.shift = shift
        self.circuit = None  # to be set by the user via ``set_circuit``

    def set_circuit(self, circuit) -> None:
        """Attach a quantum circuit wrapper that implements ``run``."""
        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.circuit is None:
            raise RuntimeError("Quantum circuit has not been attached. Call set_circuit() first.")
        x = self.classical(inputs)
        x = x.view(-1)
        return HybridFunction.apply(x, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
# Complete model: CNN backbone + hybrid head
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """
    CNN backbone that feeds into a hybrid head.  The head can be
    configured with a quantum circuit via :meth:`prepare_quantum`.
    """
    def __init__(self, depth: int = 3, shift: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # scalar output for the quantum head
        self.hybrid = Hybrid(num_features=1, depth=depth, shift=shift)

    def prepare_quantum(self, wrapper) -> None:
        """
        Attach a quantum circuit wrapper to the hybrid head.

        Parameters
        ----------
        wrapper : QuantumCircuitWrapper
            Instance that implements ``run`` and stores ``last_inputs``.
        """
        self.hybrid.set_circuit(wrapper)

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
        x = x.view(-1)  # flatten for the hybrid head
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridFunction", "Hybrid", "QCNet", "build_classical_classifier"]
