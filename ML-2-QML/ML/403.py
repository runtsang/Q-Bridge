"""Classical PyTorch implementation of the hybrid quantum binary classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics the quantum expectation."""

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


class HybridQuantumNet(nn.Module):
    """
    Classical counterpart to the hybrid quantum binary classifier.

    Architecture:
      * ResNet‑18 backbone (pre‑trained weights are not used).
      * A single linear head producing a scalar logit.
      * A differentiable sigmoid head (via HybridFunction) producing the class probability.
    """

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(inputs)
        logits = self.fc(features)
        probs = HybridFunction.apply(logits, self.shift)
        # Return two‑class probabilities
        return torch.cat((probs, 1 - probs), dim=1)


__all__ = ["HybridQuantumNet"]
