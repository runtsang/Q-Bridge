"""Hybrid sampler network combining classical self‑attention, a classical sampler,
and a differentiable hybrid head that mimics a quantum expectation layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention:
    """Simple self‑attention block implemented in PyTorch."""

    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class SamplerModule(nn.Module):
    """Feed‑forward sampler that outputs a probability distribution over two classes."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that mimics a quantum expectation value."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Linear head followed by the differentiable HybridFunction."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class HybridSamplerNet(nn.Module):
    """Hybrid network that uses classical self‑attention, a sampler, and a quantum‑style head."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.sampler = SamplerModule(input_dim=2, hidden_dim=4)
        self.hybrid = Hybrid(in_features=2, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert inputs to numpy for the attention block
        inputs_np = inputs.detach().cpu().numpy()
        # Use simple rotation and entangle params derived from the input
        rot_params = np.linspace(0, np.pi, 12)  # 4 qubits * 3 rotations
        ent_params = np.linspace(0, np.pi / 2, 3)  # 3 entangling gates
        attn_out = self.attention.run(rot_params, ent_params, inputs_np)
        # Convert back to tensor
        attn_tensor = torch.from_numpy(attn_out).float().to(inputs.device)
        # Sample probabilities
        probs = self.sampler(attn_tensor)
        # Hybrid head for classification
        logits = self.hybrid(probs)
        # Return two‑class probabilities
        return torch.cat((logits, 1 - logits), dim=-1)


__all__ = ["HybridSamplerNet"]
