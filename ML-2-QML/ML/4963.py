"""Hybrid sampler combining autoencoder, classifier, kernel, and estimator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Callable, List, Sequence

# ---------- Autoencoder ----------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(config.dropout)])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(config.dropout)])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# ---------- Kernel ----------
class _RBFKernel:
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel:
    def __init__(self, gamma: float = 1.0):
        self.ansatz = _RBFKernel(gamma)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ---------- Estimator ----------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastEstimator:
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
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

# ---------- Hybrid Sampler ----------
class HybridSamplerQNN(nn.Module):
    """Hybrid sampler network combining autoencoder encoder and classifier."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (128, 64),
                 dropout: float = 0.1,
                 kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.encoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        self.kernel = Kernel(gamma=kernel_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x)
        logits = self.classifier(z)
        return F.softmax(logits, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def compute_kernel(self,
                       a: Sequence[torch.Tensor],
                       b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, self.kernel.ansatz.gamma)

    def estimator(self, shots: int | None = None, seed: int | None = None) -> FastEstimator:
        return FastEstimator(self)

__all__ = ["HybridSamplerQNN"]
