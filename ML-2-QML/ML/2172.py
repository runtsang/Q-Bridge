"""Classical hybrid network with a ResNet18 backbone and a trainable sigmoid head.

The class is deliberately named `HybridQCNet` to mirror the quantum implementation
and to allow interchangeable usage in a joint training framework.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HybridQCNet(nn.Module):
    """
    Classical implementation mirroring the structure of the original hybrid model.
    Consists of a pretrained ResNet18 backbone, a small MLP head, and a
    differentiable sigmoid activation with optional shift.
    """

    def __init__(self, pretrained: bool = True, shift: float = 0.0, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Remove the final fully‑connected layer
        self.backbone.fc = nn.Identity()

        self.fc1 = nn.Linear(512, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 1)

        self.hybrid = HybridFunction(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


class HybridFunction(torch.autograd.Function):
    """
    Differentiable sigmoid head that optionally shifts the input before applying
    the logistic function.  It mirrors the quantum expectation head of the
    original hybrid model but remains fully classical.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


__all__ = ["HybridQCNet"]
