"""Combined classical estimator with autoencoder feature extraction and kernel ridge regression."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable, Sequence, Callable, List, Tuple

# ------------------------------------------------------------
# Autoencoder configuration and network
# ------------------------------------------------------------
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
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
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# ------------------------------------------------------------
# Kernel modules (classical RBF)
# ------------------------------------------------------------
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------
# Fast estimator utilities
# ------------------------------------------------------------
class FastBaseEstimator:
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda output: output.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
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
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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

# ------------------------------------------------------------
# Main hybrid estimator
# ------------------------------------------------------------
class EstimatorQNNGen392:
    """
    Hybrid estimator that combines a classical feed‑forward backbone,
    optional auto‑encoding feature extraction, and a kernel‑ridge regression head.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Tuple[int, int] = (8, 4),
                 use_autoencoder: bool = False,
                 autoencoder_cfg: AutoencoderConfig | None = None,
                 kernel_gamma: float = 1.0):
        # Backbone
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh(),
                  nn.Linear(hidden_dims[0], hidden_dims[1]), nn.Tanh(),
                  nn.Linear(hidden_dims[1], 1)]
        self.backbone = nn.Sequential(*layers)

        # Optional autoencoder
        self.autoencoder = None
        if use_autoencoder:
            cfg = autoencoder_cfg or AutoencoderConfig(input_dim)
            self.autoencoder = AutoencoderNet(cfg)

        # Kernel for prediction residuals
        self.kernel = Kernel(kernel_gamma)
        self._trained = False
        self._alpha: np.ndarray | None = None
        self._training_X: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, reg: float = 1e-3) -> None:
        """
        Train backbone and optional autoencoder, then compute kernel ridge coefficients.
        """
        X = X.float()
        y = y.float()

        # Store training data for prediction
        self._training_X = X.clone().detach()

        # Autoencoder training if present
        if self.autoencoder is not None:
            optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            for _ in range(50):
                recon = self.autoencoder(X)
                loss = loss_fn(recon, X)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Encode features
            with torch.no_grad():
                X = self.autoencoder.encode(X)

        # Train backbone via MSE loss
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for _ in range(200):
            preds = self.backbone(X).squeeze()
            loss = loss_fn(preds, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Compute kernel matrix for training data
        K = kernel_matrix(X, X, self.kernel.gamma)
        # Solve for alpha in ridge regression: (K + reg*I) alpha = y
        n = K.shape[0]
        self._alpha = np.linalg.solve(K + reg * np.eye(n), y.numpy())
        self._trained = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using kernel ridge formed from training data.
        """
        if not self._trained or self._alpha is None or self._training_X is None:
            raise RuntimeError("Model not fitted yet.")
        X = X.float()
        if self.autoencoder is not None:
            with torch.no_grad():
                X = self.autoencoder.encode(X)
        K_test = kernel_matrix(X, self._training_X, self.kernel.gamma)
        preds = K_test @ self._alpha
        return torch.tensor(preds)

    def evaluate(self,
                 X: torch.Tensor,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]]) -> List[List[float]]:
        """
        Wrapper around FastEstimator that applies the backbone to the inputs.
        """
        estimator = FastEstimator(self.backbone)
        return estimator.evaluate(observables, [x.tolist() for x in X])

def EstimatorQNN(*args, **kwargs):
    """Convenience wrapper matching the original anchor signature."""
    return EstimatorQNNGen392(*args, **kwargs)

__all__ = ["EstimatorQNN", "EstimatorQNNGen392"]
