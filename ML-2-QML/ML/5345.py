"""Hybrid classical network that merges classical and quantum-inspired layers.

This module builds upon the original quanvolution example and extends it with a
parameterised hybrid head that can either be a classical sigmoid layer or a
quantum expectation layer.  The class also exposes a small evaluation helper
based on the FastBaseEstimator utilities so that experiments can be run
without writing boilerplate loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FastBaseEstimator import FastBaseEstimator


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that emulates a quantum expectation value."""

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
    """Linear head followed by a quantum‑inspired sigmoid activation."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class QuanvolutionHybridNet(nn.Module):
    """Hybrid classical network combining a quanvolution filter and an optional
    quantum‑inspired head.  The architecture mirrors the original
    QuanvolutionClassifier but adds a toggle for the hybrid layer.
    """

    def __init__(
        self,
        num_classes: int = 10,
        use_quantum_head: bool = False,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.use_quantum_head = use_quantum_head
        if use_quantum_head:
            self.hybrid = Hybrid(self.linear.out_features, shift)
        else:
            self.hybrid = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Apply classical quanvolution filter
        features = self.qfilter(x)
        features = features.view(x.size(0), -1)
        logits = self.linear(features)
        if self.hybrid:
            logits = self.hybrid(logits)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables,
        parameter_sets,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ):
        """Evaluate the network for a collection of input vectors and
        observables.  The helper delegates to FastBaseEstimator so that
        experiments can be expressed in a single call.
        """
        estimator = FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["QuanvolutionHybridNet", "Hybrid", "HybridFunction"]
