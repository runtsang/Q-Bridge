"""Hybrid binary classifier with attention-gated quantum head and batched evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionMask(nn.Module):
    """Learnable mask applied to the output of the final fully connected layer."""
    def __init__(self, dim: int):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper around the quantum head using the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum, shift: float):
        ctx.quantum = quantum
        ctx.shift = shift
        exp = quantum(inputs.detach().cpu().numpy())
        result = torch.tensor(exp, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        quantum = ctx.quantum
        exp_plus = quantum((inputs + shift).detach().cpu().numpy())
        exp_minus = quantum((inputs - shift).detach().cpu().numpy())
        grad = (exp_plus - exp_minus) / 2
        grad = torch.tensor(grad, dtype=torch.float32, device=inputs.device)
        return grad * grad_output.unsqueeze(-1), None, None

class QCNet(nn.Module):
    """Hybrid CNN + quantum head with attention gating and batched quantum evaluation."""
    def __init__(self, n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        # Classical backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Attention gating
        self.attention = AttentionMask(1)

        # Quantum head
        from.quantum_module import QCNet as QuantumHead
        self.quantum = QuantumHead(n_qubits=n_qubits, shots=shots, shift=shift)
        self.shift = shift

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
        # Apply attention mask
        x = self.attention(x)
        # Quantum expectation via parameter‑shift
        q_out = HybridFunction.apply(x.squeeze(-1), self.quantum, self.shift)
        probs = torch.sigmoid(q_out).unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QCNet"]
