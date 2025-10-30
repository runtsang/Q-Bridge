"""Classical regression model with fraud‑detection inspired layers and a hybrid quantum‑like head.

The implementation combines:
- The regression dataset from `QuantumRegression.py`.
- Fraud‑detection style `FraudLayer` modules that mimic photonic layers.
- A differentiable `HybridFunction` that emulates a quantum expectation head.
- A linear head producing a scalar regression output.

The model can be trained with standard PyTorch optimisers and supports batched inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Data utilities – identical to the original `QuantumRegression.py`
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic regression data where the target is a smooth function
    of the sum of input features."""
    x = torch.rand(samples, num_features, dtype=torch.float32) * 2 - 1
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning feature tensors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": self.features[idx],
            "target": self.labels[idx],
        }

# --------------------------------------------------------------------------- #
# Fraud‑detection style layer
# --------------------------------------------------------------------------- #
class FraudLayer(nn.Module):
    """A lightweight layer inspired by the photonic fraud‑detection circuit.
    It consists of a Linear → Tanh → (scale × + shift) block.
    """
    def __init__(self, in_features: int, out_features: int, clip: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.ones(out_features))
        self.register_buffer("shift", torch.zeros(out_features))
        if clip:
            with torch.no_grad():
                self.linear.weight.clamp_(-5.0, 5.0)
                self.linear.bias.clamp_(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift

# --------------------------------------------------------------------------- #
# Hybrid quantum‑like head
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics a quantum expectation value."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class HybridHead(nn.Module):
    """Linear layer followed by the hybrid sigmoid function."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return HybridFunction.apply(logits, self.shift)

# --------------------------------------------------------------------------- #
# Main regression model
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(nn.Module):
    """Classical regression model combining fraud‑detection style layers
    and a hybrid quantum‑like head.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of fraud‑detection style layers before the head.
    """
    def __init__(self, num_features: int, depth: int = 3) -> None:
        super().__init__()
        layers = [FraudLayer(num_features, num_features, clip=False)]
        layers.extend([FraudLayer(num_features, num_features, clip=True) for _ in range(depth - 1)])
        layers.append(nn.Linear(num_features, num_features))
        self.encoder = nn.Sequential(*layers)
        self.hybrid = HybridHead(num_features)
        self.regressor = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        hybrid_out = self.hybrid(encoded)
        return self.regressor(hybrid_out).squeeze(-1)

__all__ = ["RegressionDataset", "generate_superposition_data", "QuantumRegressionModel"]
