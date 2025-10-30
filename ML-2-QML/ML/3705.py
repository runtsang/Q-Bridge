"""HybridGraphAutoencoder – classical implementation.

Combines a graph‑structured neural network with a fully‑connected autoencoder.
The class exposes classical feed‑forward, fidelity‑based graph construction,
and autoencoder training.  The quantum interface is provided externally
via the qml companion module."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn
import torch.utils.data

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input, target) pairs for supervised training."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def train_autoencoder(
    model: nn.Module,
    data: Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the autoencoder and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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


class HybridGraphAutoencoder:
    """Hybrid graph‑neural‑network + autoencoder.

    The class can operate purely classically or delegate quantum
    propagation to the corresponding qml module via the `use_quantum` flag.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        ae_config: AutoencoderConfig,
        use_quantum: bool = False,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.use_quantum = use_quantum

        # Classical weight matrices per layer
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f)
            for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:])
        ]

        # Autoencoder instance
        self.autoencoder = AutoencoderNet(ae_config)

    @property
    def target_weight(self) -> Tensor:
        return self.weights[-1]

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Classical feed‑forward through the weight matrices."""
        stored: List[List[Tensor]] = []
        for f, _ in samples:
            activations: List[Tensor] = [f]
            current = f
            for w in self.weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph based on state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            a_norm = a / (torch.norm(a) + 1e-12)
            b_norm = b / (torch.norm(b) + 1e-12)
            fid = float(torch.dot(a_norm, b_norm).item() ** 2)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train_autoencoder(
        self,
        data: Tensor,
        **kwargs,
    ) -> List[float]:
        """Delegate to the standard PyTorch autoencoder trainer."""
        return train_autoencoder(self.autoencoder, data, **kwargs)

    def random_network(self, samples: int):
        """Return training data for the target weight matrix."""
        return random_training_data(self.target_weight, samples)


__all__ = [
    "HybridGraphAutoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
