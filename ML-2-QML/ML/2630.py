"""Hybrid attention‑based binary classifier combining classical self‑attention and a dense quantum‑style head.

The network uses a convolutional backbone, a lightweight classical self‑attention block
derived from the `SelfAttention` helper, and a differentiable sigmoid head that
mimics the behaviour of a quantum expectation layer.  The design is a direct
extension of the original `QCNet` but replaces the quantum circuit with a
fully‑classical dense head, while still exposing the same interface for
plug‑in experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Self‑attention helper ----------------------------------------------------
class ClassicalSelfAttention:
    """Simple self‑attention block that mimics the interface of the quantum version."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


def SelfAttention() -> ClassicalSelfAttention:
    """Factory returning the classical self‑attention class."""
    return ClassicalSelfAttention(embed_dim=4)


# --- Hybrid head -------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that replaces the quantum expectation head."""

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
    """Dense head that emulates the behaviour of the quantum circuit."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# --- Main network -------------------------------------------------------------
class HybridAttentionQCNet(nn.Module):
    """Convolutional binary classifier with a classical self‑attention block
    and a dense hybrid head that mimics a quantum expectation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Self‑attention parameters
        self.attn_rotation = nn.Parameter(torch.randn(4 * 3))
        self.attn_entangle = nn.Parameter(torch.randn(4 - 1))
        self.self_attention = SelfAttention()

        # Hybrid head
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Self‑attention on flattened features
        rotation_np = self.attn_rotation.detach().cpu().numpy()
        entangle_np = self.attn_entangle.detach().cpu().numpy()
        attn_out = self.self_attention.run(rotation_np, entangle_np, x.detach().cpu().numpy())
        attn_out = torch.from_numpy(attn_out).to(inputs.device)

        # Feed‑forward layers
        x = F.relu(self.fc1(attn_out))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid head
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)


__all__ = ["HybridAttentionQCNet"]
