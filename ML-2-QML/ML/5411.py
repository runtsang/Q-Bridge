"""Hybrid estimator that combines classical PyTorch models with optional convolutional preprocessing."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ConvFilter(nn.Module):
    """2‑D convolution filter emulating a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class QuanvolutionFilter(nn.Module):
    """Hybrid filter that first applies a classical conv and then a quantum‑style kernel."""
    def __init__(self, kernel_size: int = 2, depth: int = 2) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size)
        self.depth = depth
        self.linear = nn.Linear(kernel_size * kernel_size, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        flat = conv_out.view(-1, self.conv.kernel_size * self.conv.kernel_size)
        return self.linear(flat)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Return a simple feed‑forward network and metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class FastBaseEstimatorGen302:
    """Hybrid estimator that supports classical PyTorch models, optional noise, and filter pipelines."""
    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
        filter: nn.Module | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set. Supports optional shot‑like noise."""
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables or [lambda out: out.mean(dim=-1)])
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                if filter is not None:
                    inputs = filter(inputs)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.to(self.device))

__all__ = ["FastBaseEstimatorGen302", "ConvFilter", "QuanvolutionFilter", "build_classifier_circuit"]
