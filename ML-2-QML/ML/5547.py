"""Hybrid convolutional neural network with kernel regularisation and graph adjacency.

The class `QCNNHybrid` extends :class:`torch.nn.Module` and combines
three key ideas from the seed projects:

* A convolution‑style stack of linear layers that mirrors the
  original QCNN architecture.
* An RBF kernel that is applied to the intermediate representations
  to compute a similarity matrix; this matrix is used to build a
  cosine‑similarity graph that can be inspected or used as a
  regularisation term.
* A lightweight estimator interface inspired by
  :class:`FastBaseEstimator` that allows evaluating the network for
  a batch of weight vectors without re‑instantiating the model.

The implementation is fully classical and depends only on
``torch`` and ``scikit‑learn`` – no external quantum libraries are
required.  The code is intentionally written so that the same
architecture can be instantiated with a quantum backend by
substituting the kernel and estimator components.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a column tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """A minimal, deterministic estimator for neural networks.

    The class mimics the behaviour of the reference implementation but
    is written in a self‑contained way so that it can be used from the
    `QCNNHybrid` class without importing the original module.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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

class QCNNHybrid(nn.Module):
    """Hybrid classical QCNN with kernel‑based regularisation.

    Architecture
    ------------
    * feature_map : 8 → 16  (Linear + ReLU)
    * conv1       : 16 → 16  (Linear + ReLU)
    * pool1       : 16 → 12  (Linear + ReLU)
    * conv2       : 12 → 8   (Linear + ReLU)
    * pool2       : 8  → 4   (Linear + ReLU)
    * conv3       : 4  → 4   (Linear + ReLU)
    * head        : 4  → 1   (Linear + Sigmoid)

    The forward pass returns the network output.  The hidden
    activations of each layer are stored in ``self._activations`` and
    can be queried via :meth:`get_activations`.

    A kernel matrix can be computed from a list of activations using
    the :meth:`kernel_matrix` method.  The similarity graph is
    constructed in :meth:`similarity_graph` using a cosine threshold.
    """
    def __init__(self, gamma: float = 1.0, similarity_thr: float = 0.9) -> None:
        super().__init__()
        self.gamma = gamma
        self.similarity_thr = similarity_thr

        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.ReLU())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 1)

        self._activations: List[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self._activations = []

        x = self.feature_map(inputs)
        self._activations.append(x)

        x = self.conv1(x)
        self._activations.append(x)

        x = self.pool1(x)
        self._activations.append(x)

        x = self.conv2(x)
        self._activations.append(x)

        x = self.pool2(x)
        self._activations.append(x)

        x = self.conv3(x)
        self._activations.append(x)

        out = torch.sigmoid(self.head(x))
        return out

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def get_activations(self) -> List[torch.Tensor]:
        """Return a copy of the stored activations."""
        return [a.clone() for a in self._activations]

    def kernel_matrix(self, activations: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute an RBF kernel matrix from a list of activations.

        Parameters
        ----------
        activations
            Iterable of tensors, each of shape ``(batch, features)``.
        """
        data = torch.cat([a.detach().cpu() for a in activations], dim=0).numpy()
        return rbf_kernel(data, gamma=self.gamma)

    def similarity_graph(self, activations: Sequence[torch.Tensor]) -> nx.Graph:
        """Construct a weighted graph from cosine similarities of activations.

        Nodes correspond to the activations of each layer.  Edges are added
        when the cosine similarity exceeds ``self.similarity_thr``.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        data = torch.cat([a.detach().cpu() for a in activations], dim=0).numpy()
        sims = cosine_similarity(data)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(activations)))

        for i in range(len(activations)):
            for j in range(i + 1, len(activations)):
                if sims[i, j] >= self.similarity_thr:
                    graph.add_edge(i, j, weight=sims[i, j])
        return graph

    # ------------------------------------------------------------------
    # Estimator utilities
    # ------------------------------------------------------------------
    def estimator(self) -> FastBaseEstimator:
        """Return a lightweight estimator for this network."""
        return FastBaseEstimator(self)

__all__ = ["QCNNHybrid", "FastBaseEstimator"]
