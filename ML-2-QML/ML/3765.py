"""Hybrid Estimator that can perform regression or binary classification.

The network is a simple fully‑connected backbone followed by either
a classical sigmoid head or a quantum expectation head.  It
combines the regression architecture from EstimatorQNN.py with the
hybrid logic from ClassicalQuantumBinaryClassification.py.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional

class HybridLayer(nn.Module):
    """
    Differentiable layer that forwards through a user supplied
    expectation function. The function receives a 1‑D tensor of
    parameters and returns a scalar expectation value.
    """
    def __init__(self, expectation_fn: Callable[[torch.Tensor], torch.Tensor],
                 shift: float = 0.0) -> None:
        super().__init__()
        self.expectation_fn = expectation_fn
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply shift before calling the expectation function
        shifted = x + self.shift
        return self.expectation_fn(shifted)

class EstimatorQNN(nn.Module):
    """
    Hybrid estimator that supports regression or binary classification.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Size of the hidden layer.
    n_outputs : int
        Number of output nodes (1 for regression, 2 for binary classification).
    use_quantum_head : bool
        If True, the last layer is a HybridLayer wrapping a quantum
        expectation circuit; otherwise a standard sigmoid head is used.
    expectation_fn : Optional[Callable]
        Function that computes a quantum expectation. Required if
        use_quantum_head=True.
    shift : float
        Bias added before the quantum layer to mimic the shift in
        HybridFunction.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_features: int = 8,
                 n_outputs: int = 1,
                 use_quantum_head: bool = False,
                 expectation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 shift: float = 0.0) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.Tanh(),
            nn.Linear(hidden_features // 2, hidden_features // 4),
            nn.Tanh(),
        )
        self.n_outputs = n_outputs
        if use_quantum_head:
            if expectation_fn is None:
                raise ValueError("expectation_fn must be provided when use_quantum_head=True")
            self.head = HybridLayer(expectation_fn, shift=shift)
        else:
            # Classic sigmoid for binary classification; linear for regression
            self.head = nn.Linear(hidden_features // 4, 1)
            if n_outputs == 2:
                self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if isinstance(self.head, HybridLayer):
            out = self.head(x)
            return out
        out = self.head(x)
        if self.n_outputs == 2:
            prob = self.sigmoid(out)
            return torch.cat([prob, 1 - prob], dim=-1)
        return out

__all__ = ["EstimatorQNN"]
