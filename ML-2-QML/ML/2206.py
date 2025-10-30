from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable activation that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class HybridLayer(nn.Module):
    """Linear layer followed by a HybridFunction."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)

class HybridEstimatorQNN(nn.Module):
    """
    Lightweight regressor that mirrors EstimatorQNN but replaces the final linear
    head with a HybridLayer, enabling a quantum‑style activation without
    requiring a quantum backend.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, shift: float = 0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.hybrid = HybridLayer(1, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.hybrid(out)

__all__ = ["HybridEstimatorQNN", "HybridLayer", "HybridFunction"]
