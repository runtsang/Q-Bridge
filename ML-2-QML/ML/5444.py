import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastHybridEstimator:
    """
    Hybrid estimator for classical PyTorch models with optional shot‑noise.
    """
    def __init__(self, model: nn.Module, *, shots: int | None = None, seed: int | None = None):
        self.model = model
        self.shots = shots
        self.seed = seed
        if shots is not None:
            self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set and observable.
        """
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
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)

        if self.shots is None:
            return results

        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    @classmethod
    def FCL(cls, n_features: int = 1):
        """
        Return a classical fully‑connected layer that mimics the quantum example.
        """
        class FullyConnectedLayer(nn.Module):
            def __init__(self, n_features: int = 1) -> None:
                super().__init__()
                self.linear = nn.Linear(n_features, 1)

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
                expectation = torch.tanh(self.linear(values)).mean(dim=0)
                return expectation.detach().numpy()
        return FullyConnectedLayer()

    @classmethod
    def Conv(cls, kernel_size: int = 2, threshold: float = 0.0):
        """
        Return a classical convolution filter that emulates a quanvolution layer.
        """
        class ConvFilter(nn.Module):
            def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.threshold = threshold
                self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

            def run(self, data) -> float:
                tensor = torch.as_tensor(data, dtype=torch.float32)
                tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
                logits = self.conv(tensor)
                activations = torch.sigmoid(logits - self.threshold)
                return activations.mean().item()
        return ConvFilter()

__all__ = ["FastHybridEstimator"]
