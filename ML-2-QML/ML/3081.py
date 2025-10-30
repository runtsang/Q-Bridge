"""ConvGen: classical convolution and graph similarity module.

This module provides a unified interface for a classical convolutional filter
and a graph-based similarity graph, inspired by the original Conv and
GraphQNN utilities.  The class can be instantiated in two distinct modes:

* ``mode='classic'`` – Uses a PyTorch Conv2d layer and a sigmoid activation.
* ``mode='graph'``   – Computes a weighted graph from a list of filter
  responses using cosine similarity.

Author: OpenAI GPT‑OSS‑20B
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import torch
from torch import nn

__all__ = ["ConvGen"]


class ConvGen:
    """Unified convolution module with classical and graph modes."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        mode: str = "classic",
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the convolution kernel.
        threshold : float
            Threshold used in the sigmoid activation.
        mode : str
            One of ``'classic'`` or ``'graph'``.
        """
        if mode not in {"classic", "graph"}:
            raise ValueError(f"Unsupported mode {mode!r}.  Only 'classic' and 'graph' are available.")
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.mode = mode

        if mode == "classic":
            # Single‑channel 2‑D convolution
            self._conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    # --------------------------------------------------------------------
    #  Classic convolution branch
    # --------------------------------------------------------------------
    def _run_classic(self, data: np.ndarray) -> float:
        """
        Run the classical convolution on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation after sigmoid.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self._conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    # --------------------------------------------------------------------
    #  Graph similarity branch
    # --------------------------------------------------------------------
    def _feature_vector(self, data: np.ndarray) -> np.ndarray:
        """
        Convert a kernel response into a flat feature vector.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            1‑D feature vector.
        """
        return data.flatten()

    def run_graph(
        self,
        data_list: list[np.ndarray],
        similarity_threshold: float = 0.8,
        *,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from a list of kernel responses.

        Parameters
        ----------
        data_list : list[np.ndarray]
            List of 2‑D arrays, each of shape (kernel_size, kernel_size).
        similarity_threshold : float
            Primary threshold for edge inclusion.
        secondary_threshold : float | None
            Secondary threshold for weighted edges.
        secondary_weight : float
            Weight assigned to secondary edges.

        Returns
        -------
        networkx.Graph
            Weighted adjacency graph.
        """
        # Compute feature vectors
        features = [self._feature_vector(data) for data in data_list]
        # Pairwise cosine similarity
        graph = nx.Graph()
        graph.add_nodes_from(range(len(features)))
        for (i, fi), (j, fj) in itertools.combinations(enumerate(features), 2):
            dot = np.dot(fi, fj)
            norm_i = np.linalg.norm(fi) + 1e-12
            norm_j = np.linalg.norm(fj) + 1e-12
            similarity = dot / (norm_i * norm_j)
            if similarity >= similarity_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary_threshold is not None and similarity >= secondary_threshold:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------
    #  Public interface
    # --------------------------------------------------------------------
    def run(self, data: np.ndarray | list[np.ndarray]) -> float | nx.Graph:
        """
        Dispatch based on the chosen mode.

        Parameters
        ----------
        data : np.ndarray | list[np.ndarray]
            Input data for the selected mode.

        Returns
        -------
        float | nx.Graph
            Result of the operation.
        """
        if self.mode == "classic":
            if not isinstance(data, np.ndarray):
                raise TypeError("Classic mode expects a single np.ndarray.")
            return self._run_classic(data)
        elif self.mode == "graph":
            if not isinstance(data, list):
                raise TypeError("Graph mode expects a list of np.ndarray.")
            return self.run_graph(data)
        else:  # pragma: no cover
            raise RuntimeError("Unsupported mode.")
