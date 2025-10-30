from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Callable, Iterable, List, Sequence

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic data for training."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
    def __len__(self): return len(self.features)
    def __getitem__(self, index: int):
        return {"states": torch.tensor(self.features[index], dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}

class ConvFilter(nn.Module):
    """Classical convolutional filter emulating a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, 1, k, k)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))  # (batch, 1)

class SamplerQNN(nn.Module):
    """Simple MLP that mimics a parameterised quantum sampler."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class FastEstimator:
    """Utility to evaluate a model over a grid of parameters with optional shot noise."""
    def __init__(self, model: nn.Module):
        self.model = model
    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in results]
        return noisy

class HybridRegressionModel(nn.Module):
    """Classical regression model that integrates a convolutional filter and a sampler head."""
    def __init__(self, num_features: int, kernel_size: int = 2):
        super().__init__()
        self.preprocess = ConvFilter(kernel_size=kernel_size)
        self.sampler = SamplerQNN()
        self.head = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch shape: (batch, num_features)
        k = int(np.sqrt(state_batch.shape[1]))
        x = state_batch.view(-1, 1, k, k)
        conv_out = self.preprocess(x)  # (batch, 1)
        samp = self.sampler(conv_out.repeat(1, 2))  # (batch, 2)
        return self.head(samp).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data",
           "ConvFilter", "SamplerQNN", "FastEstimator"]
