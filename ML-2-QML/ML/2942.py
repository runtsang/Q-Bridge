"""Classical Graph Neural Network with auto‑encoding support.

The module exposes a single :class:`GraphQNN` class that:
* builds a random feed‑forward network from a layer‑size list,
* performs forward propagation and fidelity‑based graph construction,
* uses a PyTorch fully‑connected autoencoder to compress the final
  layer into a latent representation and reconstruct it.

The API mirrors the original ``GraphQNN.py`` while adding autoencoder
capabilities derived from the ``Autoencoder.py`` seed.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn

# --------------------------------------------------------------------------- #
#  Helper functions (original seed logic retained)                           #
# --------------------------------------------------------------------------- #

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Create a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic training set for a target linear mapping."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random feed‑forward architecture."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass and return all intermediate activations."""
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
    """Squared overlap between two unit‑norm vectors."""
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
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Autoencoder integration (from Autoencoder.py)                            #
# --------------------------------------------------------------------------- #

# Import the lightweight auto‑encoder from the seed project.
from.Autoencoder import Autoencoder, AutoencoderNet, train_autoencoder


# --------------------------------------------------------------------------- #
#  GraphQNN class (public API)                                               #
# --------------------------------------------------------------------------- #

class GraphQNN:
    """
    Classical graph neural network that can embed a graph into a latent space
    using a PyTorch auto‑encoder.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes, e.g. ``[n_features, 64, 32]``.
    device : torch.device | None, optional
        Target device for tensors; defaults to CUDA if available.
    """

    def __init__(self, qnn_arch: Sequence[int], device: torch.device | None = None):
        self.arch = list(qnn_arch)
        self.arch, self.weights, self.training_data, self.target_weight = random_network(
            self.arch, samples=100
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Auto‑encoder that compresses the final hidden layer to a latent vector.
        # The latent dimension is inferred from the last weight matrix.
        self.autoencoder = Autoencoder(self.target_weight.size(1)).to(self.device)

    # --------------------------------------------------------------------- #
    #  Forward propagation utilities                                         #
    # --------------------------------------------------------------------- #

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return all activations for the provided samples."""
        return feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a weighted graph built from state fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    # --------------------------------------------------------------------- #
    #  Graph embedding & auto‑encoding                                        #
    # --------------------------------------------------------------------- #

    def _graph_to_tensor(self, graph: nx.Graph) -> Tensor:
        """Adjacency matrix as a row‑vector on the device."""
        adj = nx.to_numpy_array(graph, dtype=float)
        return torch.tensor(adj.ravel(), dtype=torch.float32, device=self.device)

    def encode_graph(self, graph: nx.Graph) -> Tensor:
        """
        Embed a graph into the latent space.

        The adjacency matrix is fed through the network, the final hidden
        layer is passed through the auto‑encoder encoder, and the latent
        representation is returned.
        """
        tensor = self._graph_to_tensor(graph)
        activations = feedforward(self.arch, self.weights, [(tensor, tensor)])
        final_hidden = activations[0][-1].unsqueeze(0)  # batch‑dim
        latent = self.autoencoder.encode(final_hidden.to(self.device))
        return latent.squeeze(0)

    def decode_latent(self, latent: Tensor) -> Tensor:
        """
        Reconstruct the original graph representation from a latent vector.

        The latent is decoded by the auto‑encoder and the result is reshaped
        into the adjacency matrix shape.
        """
        latent = latent.unsqueeze(0).to(self.device)
        recon = self.autoencoder.decode(latent)
        dim = int(self.arch[0] ** 0.5)
        return recon.squeeze(0).reshape(dim, dim)

    # --------------------------------------------------------------------- #
    #  Auto‑encoder training                                                 #
    # --------------------------------------------------------------------- #

    def train_autoencoder(
        self,
        data: Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> List[float]:
        """
        Train the embedded auto‑encoder on the provided data.

        Parameters
        ----------
        data : Tensor
            Training data of shape ``(N, n_features)``.
        """
        return train_autoencoder(
            self.autoencoder,
            data.to(self.device),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
        )

    # --------------------------------------------------------------------- #
    #  Utility accessors                                                    #
    # --------------------------------------------------------------------- #

    @property
    def weight_matrix(self) -> Tensor:
        """Return the final linear layer weight matrix."""
        return self.weights[-1]

    def __repr__(self) -> str:
        return f"<GraphQNN arch={self.arch} device={self.device}>"

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
