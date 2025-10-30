"""Hybrid classical-quantum binary classifier.

The `HybridClassifier` class implements a convolutional feature extractor,
dense layers, a classical autoencoder, and a quantum‑inspired head that
emulates a parameterised quantum circuit using a tanh‑based differentiable
function.  It mirrors the architecture of the original quantum model while
remaining fully classical and end‑to‑end trainable on CPU/GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Autoencoder import Autoencoder
from FCL import FCL


class HybridFunction(torch.autograd.Function):
    """Differentiable quantum‑inspired activation using tanh with a shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        out = torch.tanh(inputs + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        grad = grad_output * (1 - out ** 2)
        return grad, None


class Hybrid(nn.Module):
    """Classical dense head that emulates the quantum expectation layer."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return HybridFunction.apply(logits, self.shift)


class HybridClassifier(nn.Module):
    """End‑to‑end hybrid binary classifier."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 1)
        self.drop = nn.Dropout2d(0.3)

        # Dense layers
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)  # latent dimension

        # Classical autoencoder
        self.autoenc = Autoencoder(
            input_dim=32,
            latent_dim=16,
            hidden_dims=(64, 32),
            dropout=0.1,
        )

        # Fully connected layer approximating the quantum head
        self.fcl = FCL()()  # instantiate the fully connected layer

        # Quantum‑inspired head
        self.hybrid = Hybrid(32, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)

        # Dense transformation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 32‑dim latent

        # Autoencoder compression
        z = self.autoenc.encode(x)

        # Quantum‑inspired head
        q_out = self.hybrid(z)

        # Classical FCL output
        fcl_out = torch.tensor(
            self.fcl.run(z.squeeze().tolist()),
            device=x.device,
            dtype=x.dtype,
        )

        # Final probability (two‑class)
        prob = torch.sigmoid(q_out + fcl_out)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["HybridClassifier"]
