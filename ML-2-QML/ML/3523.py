import torch
from torch import nn
import numpy as np
from typing import Iterable, Callable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class HybridConvEstimator(nn.Module):
    """
    Classical implementation of a convolutional filter that mimics the behaviour of the
    original Conv class but is fully PyTorch‑based.  It can be used as a drop‑in
    replacement for the quantum filter in a hybrid pipeline.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 bias: float = 0.0) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square filter.  The module expects input tensors of shape
            (N, 1, H, W) where H and W are at least kernel_size.
        threshold : float
            Value subtracted from the convolution logits before the sigmoid.
        bias : float
            Optional constant bias added to the convolution weights.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # Initialise bias to the provided value for reproducibility
        torch.nn.init.constant_(self.conv.bias, bias)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns the mean sigmoid‑activated convolution output.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (N, 1, H, W) or (1, 1, k, k).

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the mean activation across the batch.
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def evaluate(self,
                 observables: Iterable[ScalarObservable] | None = None,
                 parameter_sets: Sequence[Sequence[float]] | None = None) -> List[List[float]]:
        """
        Evaluate the filter for a list of parameter tuples.  Each parameter set
        is interpreted as the flattened weight vector for the convolution
        followed by the bias term.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Optional scalar observables applied to the model output.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter vectors.  Each vector must have length
            ``kernel_size**2 + 1`` (weights + bias).

        Returns
        -------
        List[List[float]]
            For each parameter set a list of observable values.
        """
        if observables is None:
            observables = [lambda x: x.mean()]
        if parameter_sets is None:
            return []

        results: List[List[float]] = []
        # Simple loop – no batching for clarity
        for params in parameter_sets:
            # Unpack parameters into weights and bias
            weight_vec, bias_val = params[:-1], params[-1]
            weight_tensor = torch.tensor(weight_vec, dtype=torch.float32).view(1, 1,
                                                                             self.kernel_size,
                                                                             self.kernel_size)
            self.conv.weight.data = weight_tensor
            self.conv.bias.data = torch.tensor([bias_val], dtype=torch.float32)
            # Dummy input: a single kernel‑sized patch
            dummy = torch.zeros((1, 1, self.kernel_size, self.kernel_size))
            output = self.forward(dummy)
            row = []
            for obs in observables:
                val = obs(output)
                if isinstance(val, torch.Tensor):
                    row.append(float(val.mean().cpu()))
                else:
                    row.append(float(val))
            results.append(row)
        return results

__all__ = ["HybridConvEstimator"]
