"""Hybrid estimator that unifies classical and quantum primitives.

This module defines :class:`HybridEstimator` which extends the lightweight
FastEstimator from the seed project.  It can optionally wrap a convolution
filter, an auto‑encoder and a classifier, allowing a fully classical
pipeline that mirrors the quantum workflow in the QML seed.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# Base estimator copied from the seed (FastEstimator)
# --------------------------------------------------------------------------- #
class FastEstimator:
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        raw = self._evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def _evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
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

# --------------------------------------------------------------------------- #
# Auxiliary components (Conv, Autoencoder, Classifier)
# --------------------------------------------------------------------------- #
# Conv filter (classical)
class ClassicalConv:
    """Simple 2‑D convolution filter implemented in PyTorch."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# Auto‑encoder (simple MLP)
class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# Classifier factory (simple feed‑forward)
def build_classifier_circuit(num_features: int, depth: int) -> nn.Module:
    layers = []
    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    return nn.Sequential(*layers)

# --------------------------------------------------------------------------- #
# HybridEstimator
# --------------------------------------------------------------------------- #
class HybridEstimator(FastEstimator):
    """Estimator that chains optional convolution, auto‑encoder and
    classifier modules.  It inherits the shot‑noise behaviour of
    :class:`FastEstimator` and adds preprocessing capabilities.
    """
    def __init__(
        self,
        model: nn.Module,
        conv: Optional[Callable] = None,
        autoencoder: Optional[AutoencoderNet] = None,
        classifier: Optional[nn.Module] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(model)
        self.conv = conv
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.shots = shots
        self.seed = seed

    def _preprocess(self, data_batch: torch.Tensor) -> torch.Tensor:
        """Apply optional conv and auto‑encoder transforms."""
        if self.conv is not None:
            conv_out = []
            for sample in data_batch:
                conv_out.append(self.conv.run(sample.cpu().numpy()))
            data_batch = torch.tensor(conv_out, dtype=torch.float32, device=data_batch.device)

        if self.autoencoder is not None:
            encoded = self.autoencoder.encode(data_batch)
            decoded = self.autoencoder.decode(encoded)
            data_batch = decoded

        return data_batch

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Evaluate the wrapped model for a list of parameter sets."""
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                tensor = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                batch = self._preprocess(tensor)
                outputs = self.model(batch)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        # Add shot noise if requested
        if shots is None:
            shots = self.shots
        if shots is None:
            return results

        rng = np.random.default_rng(seed if seed is not None else self.seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> List[int]:
        """Return class labels using the optional classifier head."""
        self.model.eval()
        predictions: List[int] = []
        with torch.no_grad():
            for params in parameter_sets:
                tensor = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                batch = self._preprocess(tensor)
                logits = self.model(batch)
                if self.classifier is not None:
                    logits = self.classifier(logits)
                pred = int(torch.argmax(logits, dim=-1).item())
                predictions.append(pred)
        return predictions

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> List[float]:
        """Convenience wrapper that trains the internal auto‑encoder."""
        if self.autoencoder is None:
            raise RuntimeError("No auto‑encoder supplied to this estimator.")
        return train_autoencoder(
            self.autoencoder,
            data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

__all__ = ["HybridEstimator"]
