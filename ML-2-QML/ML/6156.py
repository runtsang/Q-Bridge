import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical hybrid head mimicking quantum expectation
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
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

class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridEstimatorQNN(nn.Module):
    """
    A lightweight regression/classification backbone that optionally
    delegates the final activation to a classical or quantum head.
    """
    def __init__(self, num_features: int = 2, use_quantum: bool = False,
                 n_qubits: int = 2, backend=None, shots: int = 100,
                 shift: float = 0.0, quantum_head=None) -> None:
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the input vector.
        use_quantum : bool
            If True, expects a quantum_head callable; otherwise uses
            the classical Hybrid head.
        n_qubits, backend, shots, shift : optional
            Passed to the quantum head when ``use_quantum`` is True.
        quantum_head : callable, optional
            A callable that accepts a tensor and returns a tensor.
            If provided, overrides the default quantum head.
        """
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        if use_quantum:
            if quantum_head is None:
                raise ValueError("When use_quantum is True, a quantum_head callable must be supplied.")
            self.head = quantum_head
        else:
            self.head = Hybrid(self.core.out_features, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.core(x)
        return self.head(features)

def EstimatorQNN() -> HybridEstimatorQNN:
    """
    Factory function that mirrors the original EstimatorQNN signature.
    Returns a default classical estimator suitable for regression.
    """
    return HybridEstimatorQNN(num_features=2, use_quantum=False)

__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
