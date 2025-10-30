from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

def _batchify(values: Sequence[float]) -> torch.Tensor:
    """Coerce a sequence of floats into a 2‑D float32 tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridFastEstimator:
    """Unified estimator for PyTorch models and quantum circuits.

    When instantiated with an ``nn.Module`` the class behaves like the
    original :class:`FastBaseEstimator`, supporting arbitrary scalar
    observables and optional Gaussian shot noise.  When given a
    ``Callable[[torch.Tensor], torch.Tensor]`` it treats the callable
    as a forward pass and evaluates it on batched inputs.
    """

    def __init__(self, model: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.model = model

    def _forward(self, params: Sequence[float]) -> torch.Tensor:
        inputs = _batchify(params)
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                return self.model(inputs)
        else:
            return self.model(inputs)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables
            Callables that map a model output tensor to a scalar or a
            tensor that can be reduced to a scalar.
        parameter_sets
            Iterable of parameter sequences to evaluate.
        shots
            If supplied, a Gaussian noise with variance ``1/shots`` is
            added to each mean value.
        seed
            Seed for reproducible noise.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        for params in parameter_sets:
            outputs = self._forward(params)
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

class SamplerQNN(nn.Module):
    """A lightweight neural sampler mirroring the Qiskit ``SamplerQNN``."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.softmax(self.net(inputs), dim=-1)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> nn.Module:
    """Construct a small fully‑connected auto‑encoder used as a toy model."""
    class _Config:
        def __init__(self, input_dim: int, latent_dim: int, hidden_dims: tuple[int, int], dropout: float):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.hidden_dims = hidden_dims
            self.dropout = dropout

    cfg = _Config(input_dim, latent_dim, hidden_dims, dropout)

    layers = []
    in_dim = cfg.input_dim
    for h in cfg.hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        if cfg.dropout > 0.0:
            layers.append(nn.Dropout(cfg.dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, cfg.latent_dim))
    encoder = nn.Sequential(*layers)

    layers = []
    in_dim = cfg.latent_dim
    for h in reversed(cfg.hidden_dims):
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        if cfg.dropout > 0.0:
            layers.append(nn.Dropout(cfg.dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, cfg.input_dim))
    decoder = nn.Sequential(*layers)

    class _Autoencoder(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return decoder(encoder(x))

    return _Autoencoder()

def train_autoencoder(
    model: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple training loop that returns the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridFastEstimator",
    "SamplerQNN",
    "Autoencoder",
    "train_autoencoder",
]
