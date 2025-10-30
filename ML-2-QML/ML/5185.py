import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected / convolutional layer.

    Parameters
    ----------
    mode : str
        ``'fc'`` for a classic fully‑connected layer, ``'conv'`` for a 2‑D
        convolutional filter.  The implementation is a drop‑in replacement
        for the quantum layers in the reference seeds.
    n_features : int, optional
        Number of input features for the fully‑connected mode.
    kernel_size : int, optional
        Size of the 2‑D kernel for the convolutional mode.
    threshold : float, optional
        Threshold used in the sigmoid activation of the conv mode.
    """
    def __init__(self,
                 mode: str = "fc",
                 n_features: int = 1,
                 kernel_size: int = 2,
                 threshold: float = 0.0) -> None:
        super().__init__()
        self.mode = mode
        if mode == "fc":
            self.layer = nn.Linear(n_features, 1)
        elif mode == "conv":
            self.layer = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.threshold = threshold
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fc":
            return torch.tanh(self.layer(x)).mean(dim=0)
        else:
            if x.ndim == 2:
                tensor = x.view(1, 1, *x.shape)
            else:
                tensor = x
            logits = self.layer(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the quantum API: accept a list of parameters and return a
        single‑element NumPy array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            output = self.forward(values)
        return np.array([output.item()])

class FastBaseEstimator:
    """Evaluate a model for batches of parameters and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""
    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridFCL", "FastBaseEstimator", "FastEstimator"]
