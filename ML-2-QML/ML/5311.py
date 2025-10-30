"""GraphQNNGen197: hybrid classical graph neural network with optional autoencoder and fully‑connected layer."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Tuple as TupleType

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Auxiliary helpers ---------------------------------------------------------
from Autoencoder import Autoencoder, AutoencoderConfig, AutoencoderNet
from FCL import FCL

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Randomly initialise a weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (x, Wx) for a target weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight matrix chain and corresponding training data."""
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
    """Forward pass through a classical feed‑forward network."""
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
    """Squared overlap of two feature vectors."""
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
    """Build a weighted graph from state‑fidelity similarities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --- Main hybrid class ---------------------------------------------------------
class GraphQNNGen197:
    """Unified interface for classical graph‑neural‑network experiments.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the feed‑forward network.
    autoencoder : bool, optional
        If True, an AutoencoderNet is instantiated and can be used for feature compression.
    fcl : bool, optional
        If True, a simple fully‑connected layer (FCL) is added for experimentation.
    latent_dim, hidden_dims, dropout : optional
        Hyper‑parameters for the autoencoder when enabled.
    """

    def __init__(
        self,
        arch: Sequence[int],
        *,
        autoencoder: bool = False,
        fcl: bool = False,
        latent_dim: int = 32,
        hidden_dims: TupleType[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.arch = list(arch)
        self.autoencoder = autoencoder
        self.fcl = fcl

        # Autoencoder setup
        self.autoencoder_cfg = (
            AutoencoderConfig(
                input_dim=self.arch[0],
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            if autoencoder
            else None
        )
        self.autoencoder_model: AutoencoderNet | None = (
            Autoencoder(
                input_dim=self.autoencoder_cfg.input_dim,
                latent_dim=self.autoencoder_cfg.latent_dim,
                hidden_dims=self.autoencoder_cfg.hidden_dims,
                dropout=self.autoencoder_cfg.dropout,
            )
            if autoencoder
            else None
        )

        # Fully‑connected layer
        self.fcl_model: nn.Module | None = FCL() if fcl else None

        self.weights: List[Tensor] | None = None

    # -------------------------------------------------------------------------
    def random_network(self, samples: int):
        """Generate a random network and training data."""
        arch, weights, training_data, target_weight = random_network(self.arch, samples)
        self.weights = weights
        return arch, weights, training_data, target_weight

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]):
        """Run samples through the current network."""
        if self.weights is None:
            raise RuntimeError("Weights not initialised; call random_network() first.")
        return feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ):
        """Return a graph built from state fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    # -------------------------------------------------------------------------
    def autoencode(self, data: Tensor) -> Tensor:
        """Encode data using the autoencoder (if enabled)."""
        if not self.autoencoder or self.autoencoder_model is None:
            raise RuntimeError("Autoencoder is not configured.")
        return self.autoencoder_model.encode(data)

    def fcl_run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Run a fully‑connected layer on provided parameters."""
        if not self.fcl or self.fcl_model is None:
            raise RuntimeError("FCL is not configured.")
        return torch.tensor(self.fcl_model.run(thetas))
