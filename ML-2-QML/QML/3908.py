"""HybridClassifier – quantum PyTorch implementation.

The quantum model uses a two‑qubit SamplerQNN as the hybrid head.
It can be replaced by a simple expectation‑based layer if desired.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler as StatevectorSampler


class QuantumSampler(nn.Module):
    """Two‑qubit SamplerQNN built from a ParameterVector circuit."""

    def __init__(self, shots: int = 1024) -> None:
        super().__init__()
        self.shots = shots

        # Build a parameterised circuit
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        self.base_circuit = QuantumCircuit(2)
        self.base_circuit.ry(self.inputs[0], 0)
        self.base_circuit.ry(self.inputs[1], 1)
        self.base_circuit.cx(0, 1)
        self.base_circuit.ry(self.weights[0], 0)
        self.base_circuit.ry(self.weights[1], 1)
        self.base_circuit.cx(0, 1)
        self.base_circuit.ry(self.weights[2], 0)
        self.base_circuit.ry(self.weights[3], 1)

        # Trainable weight parameters
        self.weight_params = nn.Parameter(torch.randn(4))

        # Sampler primitive (state‑vector based)
        self.sampler = StatevectorSampler()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: Tensor of shape (batch, 2) – the two input angles.
        Returns a probability of measuring the |00⟩ state.
        """
        batch_size = inputs.shape[0]
        probs = []

        for i in range(batch_size):
            # Bind input angles and current trainable weights
            bound_params = {
                self.inputs[j]: inputs[i, j].item() for j in range(2)
            }
            bound_params.update({
                self.weights[j]: self.weight_params[j].item() for j in range(4)
            })
            bound_circuit = self.base_circuit.assign_parameters(bound_params, inplace=False)

            # Generate the statevector and compute probabilities
            state = Statevector.from_instruction(bound_circuit)
            prob_dict = state.probabilities_dict()
            prob_00 = prob_dict.get("00", 0.0)
            probs.append(prob_00)

        probs_tensor = torch.tensor(probs, device=inputs.device, dtype=torch.float32)
        return probs_tensor


class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the QuantumSampler."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, sampler: QuantumSampler, shift: float = 0.0) -> torch.Tensor:
        ctx.shift = shift
        ctx.sampler = sampler
        # Apply shift to the inputs for finite‑difference gradient
        shifted = inputs + shift
        # Run the quantum sampler (no autograd support)
        probs = ctx.sampler(shifted)
        ctx.save_for_backward(inputs, probs)
        return probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, probs = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift

        # Finite‑difference approximation of gradients
        grads = []
        for idx, (inp, sh) in enumerate(zip(inputs.tolist(), shift)):
            right = ctx.sampler(torch.tensor(inp + sh))
            left = ctx.sampler(torch.tensor(inp - sh))
            grads.append((right - left).item())

        grad_inputs = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """Quantum hybrid head that forwards activations through the sampler."""

    def __init__(self, shift: float = 0.0, shots: int = 1024) -> None:
        super().__init__()
        self.sampler = QuantumSampler(shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.sampler, self.shift)


class QCNet(nn.Module):
    """CNN backbone followed by a quantum hybrid head."""

    def __init__(self, shift: float = 0.0, shots: int = 1024) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.hybrid = Hybrid(shift=shift, shots=shots)

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
        logits = self.fc3(x)

        probs = self.hybrid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumSampler", "HybridFunction", "Hybrid", "QCNet"]
