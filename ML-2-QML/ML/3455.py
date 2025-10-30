import torch
from torch import nn
import numpy as np
from typing import Callable, Iterable, List, Sequence

class HybridConvolution(nn.Module):
    """
    Classical convolutional filter with optional shot‑noise simulation.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 shots: int | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.shots = shots
        self.seed = seed
        if shots is not None:
            self.rng = np.random.default_rng(seed)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean sigmoid activation of the convolution output.
        """
        x = data.view(1, 1, self.kernel_size, self.kernel_size).float()
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        mean_val = activations.mean()
        if self.shots is not None:
            noise = self.rng.normal(0, max(1e-6, 1 / self.shots))
            mean_val = mean_val + noise
        return mean_val

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """
        Evaluate a list of observables for each parameter set.
        Each parameter set is a list of floats that will be assigned to the
        convolutional weights and bias in order.
        """
        observables = list(observables) or [lambda x: x.mean()]
        results: List[List[float]] = []
        for params in parameter_sets:
            with torch.no_grad():
                # Set convolution weights
                weight_vals = params[:self.conv.weight.numel()]
                bias_vals = params[self.conv.weight.numel():]
                if weight_vals:
                    weight = torch.tensor(weight_vals,
                                          dtype=self.conv.weight.dtype,
                                          device=self.conv.weight.device).view_as(self.conv.weight)
                    self.conv.weight.copy_(weight)
                if bias_vals:
                    bias = torch.tensor(bias_vals,
                                        dtype=self.conv.bias.dtype,
                                        device=self.conv.bias.device)
                    self.conv.bias.copy_(bias)
                # Dummy data: zeros
                dummy = torch.zeros((self.kernel_size, self.kernel_size))
                out = self.forward(dummy)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    row.append(float(val))
                results.append(row)
        return results
