"""FastHybridEstimator: classical implementation combining neural nets, autoencoders, and graph QNN utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List, Optional
import numpy as np
import torch
from torch import nn
import networkx as nx


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D float32 tensor, adding a batch dimension if necessary."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Cast data to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# Autoencoder components
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder with optional dropout."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Convenience factory mirroring the quantum helper."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> list[float]:
    """Simple training loop returning a loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
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


# --------------------------------------------------------------------------- #
# Graph‑based QNN utilities (classical)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Unified estimator
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """
    A versatile estimator that can:
    * evaluate a PyTorch neural network or auto‑encoder on batches of inputs,
    * compute graph‑based state fidelities and build adjacency graphs,
    * optionally add Gaussian shot noise to deterministic predictions.
    """
    def __init__(
        self,
        model: nn.Module | None = None,
        autoencoder: AutoencoderNet | None = None,
        graph_arch: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.autoencoder = autoencoder
        self.graph_arch = graph_arch

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is not None:
            return self.autoencoder(inputs)
        if self.model is not None:
            return self.model(inputs)
        raise RuntimeError("No model or auto‑encoder configured.")

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Batch‑evaluate the underlying model/auto‑encoder and return a list of scalar
        observables for each input.  If ``shots`` is provided, Gaussian noise with
        variance ``1/shots`` is added to emulate finite‑sampling.
        """
        if not parameter_sets:
            return []

        observables = list(observables or [lambda out: out.mean(dim=-1)])
        results: List[List[float]] = []
        self.model.eval() if self.model else None
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self._forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode data using the auto‑encoder (if present)."""
        if self.autoencoder is None:
            raise RuntimeError("Auto‑encoder not configured.")
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation using the auto‑encoder."""
        if self.autoencoder is None:
            raise RuntimeError("Auto‑encoder not configured.")
        return self.autoencoder.decode(z)

    def graph_fidelity(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
    ) -> nx.Graph:
        """Return a fidelity‑based adjacency graph for a set of state vectors."""
        return fidelity_adjacency(states, threshold, secondary=secondary)


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "FastHybridEstimator",
]
