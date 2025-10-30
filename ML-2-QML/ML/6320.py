"""Hybrid graph neural network and autoencoder utilities.

This module merges the classical GraphQNN utilities with a lightweight
autoencoder.  The public API mirrors the original GraphQNN module
(`feedforward`, `fidelity_adjacency`, `random_network`, `random_training_data`,
`state_fidelity`) while adding a fully‑connected Autoencoder that can be
trained independently or used to generate a latent‑space graph.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Classical autoencoder
# --------------------------------------------------------------------------- #

class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Light‑weight fully‑connected autoencoder."""
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

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    """Factory that mirrors the quantum helper."""
    return AutoencoderNet(cfg)


def train_autoencoder(model: AutoencoderNet,
                      data: Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> List[float]:
    """Simple reconstruction training loop returning the loss history."""
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
#  Graph utilities (from the original GraphQNN)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (x, Wx) pairs for a linear target."""
    return [(torch.randn(weight.size(1), dtype=torch.float32),
             weight @ torch.randn(weight.size(1), dtype=torch.float32))
            for _ in range(samples)]


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weight list, training data, and target weight."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[Tensor],
                samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Return the activations for each sample through the network."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        act: List[Tensor] = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            act.append(current)
        activations.append(act)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
#  Hybrid class
# --------------------------------------------------------------------------- #

class GraphQNNAutoencoder:
    """Hybrid classical GraphQNN + autoencoder.

    The class keeps a small feed‑forward network (defined by ``qnn_arch``)
    and a fully‑connected autoencoder.  It exposes the original GraphQNN
    utilities together with convenience methods for training the
    autoencoder and building a latent‑space graph.
    """

    def __init__(self,
                 qnn_arch: Sequence[int],
                 autoencoder_cfg: AutoencoderConfig) -> None:
        self.qnn_arch = list(qnn_arch)
        self.autoencoder = Autoencoder(autoencoder_cfg)
        self.weights = [ _random_linear(in_f, out_f)
                         for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]) ]

    # ----- classical GraphQNN helpers ------------------------------------------------

    def feedforward(self,
                    samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations for each sample."""
        return feedforward(self.qnn_arch, self.weights, samples)

    def get_graph_from_fidelities(self,
                                  states: Sequence[Tensor],
                                  threshold: float,
                                  *,
                                  secondary: float | None = None,
                                  secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

    # ----- autoencoder helpers -------------------------------------------------------

    def train_autoencoder(self,
                          data: Tensor,
                          **kwargs) -> List[float]:
        """Train the internal autoencoder and return loss history."""
        return train_autoencoder(self.autoencoder, data, **kwargs)

    def encode_latents(self,
                       samples: Iterable[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        """Return latent vectors for the given input samples."""
        return [self.autoencoder.encode(inp) for inp, _ in samples]

    def build_latent_graph(self,
                           samples: Iterable[Tuple[Tensor, Tensor]],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Construct a graph from latent vectors using the fidelity threshold."""
        latents = self.encode_latents(samples)
        return self.get_graph_from_fidelities(latents,
                                              threshold,
                                              secondary=secondary,
                                              secondary_weight=secondary_weight)

    # ----- static helpers ------------------------------------------------------------

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Proxy to the original random_network."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        """Proxy to the original random_training_data."""
        return random_training_data(weight, samples)

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
    "GraphQNNAutoencoder",
]
