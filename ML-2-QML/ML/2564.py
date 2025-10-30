"""Hybrid graph autoencoder combining classical GNN layers with an MLP autoencoder.

The module re‑implements the core utilities from the original GraphQNN
(seed) and augments them with a fully‑connected autoencoder.  The
`GraphQNN_AE` class exposes a single forward method that:
1. propagates node features through a sequence of linear layers
   (classical GNN part),
2. takes the last layer as a latent vector,
3. decodes it back to the feature space via the autoencoder.
The class also provides a helper to compute a fidelity‑based adjacency
graph from the latent states.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Iterable as IterableType, Sequence as SequenceType, Tuple as TupleType

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1. Classical GraphQNN utilities (adapted from seed)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight list and synthetic training data."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass through the classical GNN part."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
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
# 2. Autoencoder utilities (adapted from seed)
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class AutoencoderConfig:
    """Configuration for the MLP autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Standard fully‑connected autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
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

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory mirroring the quantum helper."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


# --------------------------------------------------------------------------- #
# 3. Hybrid classical GraphQNN + Autoencoder
# --------------------------------------------------------------------------- #
class GraphQNN_AE(nn.Module):
    """
    Hybrid graph neural network autoencoder that:
    * propagates node features through a sequence of linear layers (classical GNN),
    * uses the last layer as a latent representation,
    * decodes the latent vector with a standard MLP autoencoder.
    The class also exposes a helper to compute a fidelity‑based adjacency graph
    from the latent states.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        autoencoder_cfg: AutoencoderConfig,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        ]
        self.autoencoder = Autoencoder(
            input_dim=autoencoder_cfg.input_dim,
            latent_dim=autoencoder_cfg.latent_dim,
            hidden_dims=autoencoder_cfg.hidden_dims,
            dropout=autoencoder_cfg.dropout,
        )

    def encode_graph(self, node_features: Tensor) -> List[Tensor]:
        """Forward pass through the classical GNN part."""
        activations = [node_features]
        current = node_features
        for weight in self.weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        return activations

    def reconstruct_features(self, latent: Tensor) -> Tensor:
        """Decode latent representation back to node feature space."""
        return self.autoencoder.decode(latent)

    def forward(self, node_features: Tensor) -> Tensor:
        """
        Full forward: GNN encoding → latent via autoencoder → reconstruction.
        The output is the reconstructed node features.
        """
        gnn_activations = self.encode_graph(node_features)
        latent = gnn_activations[-1]
        recon = self.reconstruct_features(latent)
        return recon

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None) -> nx.Graph:
        """
        Build a graph where nodes are the latent states from the GNN part
        and edges are weighted by fidelity between latent vectors.
        """
        # Dummy node to compute latent states; in practice use real graph features
        dummy_features = torch.randn(self.qnn_arch[0])
        states = self.encode_graph(dummy_features)
        return fidelity_adjacency(states, threshold, secondary=secondary)


__all__ = [
    "GraphQNN_AE",
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "train_autoencoder",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
