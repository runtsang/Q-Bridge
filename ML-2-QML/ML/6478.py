from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import torch
import networkx as nx
import numpy as np

Tensor = torch.Tensor


class GraphQNNHybrid:
    """
    Classical implementation of a graph neural network with an optional
    PyTorch autoencoder.  Methods mirror the original GraphQNN utilities
    but are encapsulated in a single class that can be instantiated
    with a target architecture and an autoencoder configuration.
    """

    def __init__(self, qnn_arch: Sequence[int], autoencoder_config: dict | None = None):
        """
        Parameters
        ----------
        qnn_arch : Sequence[int]
            Layer widths of the linear feed‑forward network.
        autoencoder_config : dict, optional
            Configuration dictionary compatible with :class:`Autoencoder`.
        """
        self.qnn_arch = list(qnn_arch)
        self.autoencoder_config = autoencoder_config or {}
        self.weights = self._init_random_weights()
        self.autoencoder = None
        if self.autoencoder_config:
            from.Autoencoder import Autoencoder
            self.autoencoder = Autoencoder(**self.autoencoder_config)

    def _init_random_weights(self) -> List[Tensor]:
        """Generate a list of random weight matrices."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        return weights

    def feedforward(self, features: Tensor) -> List[List[Tensor]]:
        """
        Forward pass through the linear network using tanh activations.

        Returns
        -------
        activations : List[List[Tensor]]
            Layer‑wise activations including the input layer.
        """
        activations = [features]
        current = features
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    def random_training_data(self,
                             target_weight: Tensor,
                             samples: int) -> List[Tuple[Tensor, Tensor]]:
        """
        Generate synthetic training data for a target linear layer.

        Parameters
        ----------
        target_weight : Tensor
            The weight matrix to emulate.
        samples : int
            Number of data points.

        Returns
        -------
        dataset : List[Tuple[Tensor, Tensor]]
            (input, target) pairs.
        """
        dataset = []
        for _ in range(samples):
            x = torch.randn(target_weight.size(1), dtype=torch.float32)
            y = target_weight @ x
            dataset.append((x, y))
        return dataset

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor],
                                                    List[Tuple[Tensor, Tensor]],
                                                    Tensor]:
        """
        Generate a random network, training data, and the target weight.

        Returns
        -------
        arch : List[int]
            Architecture list.
        weights : List[Tensor]
            Randomly initialized weights.
        training_data : List[Tuple[Tensor, Tensor]]
            Synthetic dataset for the final layer.
        target_weight : Tensor
            The final layer weight matrix.
        """
        target_weight = self.weights[-1]
        training_data = self.random_training_data(target_weight, samples)
        return self.qnn_arch, self.weights, training_data, target_weight

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two normalized vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted adjacency graph from state fidelities.

        Parameters
        ----------
        states : Sequence[Tensor]
            List of state vectors.
        threshold : float
            Primary fidelity threshold.
        secondary : float, optional
            Secondary threshold for a lighter edge weight.
        secondary_weight : float, default 0.5
            Weight assigned to secondary edges.

        Returns
        -------
        graph : nx.Graph
            Weighted graph of state similarities.
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    def train_autoencoder(self,
                           data: torch.Tensor,
                           epochs: int = 100,
                           batch_size: int = 64,
                           lr: float = 1e-3,
                           weight_decay: float = 0.0,
                           device: torch.device | None = None) -> List[float]:
        """
        Train the optional autoencoder on supplied data.

        Parameters
        ----------
        data : torch.Tensor
            Input data to reconstruct.
        epochs, batch_size, lr, weight_decay, device : see :func:`train_autoencoder`.

        Returns
        -------
        history : List[float]
            Training loss per epoch.
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not configured.")
        from.Autoencoder import train_autoencoder
        return train_autoencoder(self.autoencoder, data,
                                 epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=weight_decay,
                                 device=device)


__all__ = ["GraphQNNHybrid"]
