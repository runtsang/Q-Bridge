"""Hybrid SelfAttention with graph‑based QNN utilities.

This module merges the classical self‑attention logic from the original
SelfAttention seed with the graph‑based utilities from GraphQNN.  It
provides a single class SelfAttention that can be used as a pure neural
network layer or as a helper to build a weighted adjacency graph
from the activations of a quantum‑enhanced feed‑forward pass.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Iterable as IterableType

import networkx as nx
import torch
import numpy as np

Tensor = torch.Tensor
Array = np.ndarray

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs for the target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

class SelfAttention:
    """Hybrid self‑attention + graph‑based QNN.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    qnn_arch : Sequence[int]
        Architecture of the quantum‑neural‑network (number of qubits per layer).
    """

    def __init__(self, embed_dim: int, qnn_arch: Sequence[int] = (4, 4, 4)):
        self.embed_dim = embed_dim
        self.qnn_arch = list(qnn_arch)

        # Classical self‑attention weights
        self.Wq = _random_linear(embed_dim, embed_dim)
        self.Wk = _random_linear(embed_dim, embed_dim)
        self.Wv = _random_linear(embed_dim, embed_dim)

        # Quantum‑neural‑network state‑propagation data
        self._qnn_weights: List[List[Tensor]] | None = None
        self._qnn_training_data: List[Tuple[Tensor, Tensor]] | None = None

    # ------------------------------------------------------------------ #
    #  Classical self‑attention
    # ------------------------------------------------------------------ #
    def run(self, inputs: Array) -> Array:
        """Compute a classical self‑attention map.

        Parameters
        ----------
        inputs : np.ndarray shape (batch, seq_len, embed_dim)
            Input embeddings.
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        query = inputs_t @ self.Wq
        key = inputs_t @ self.Wk
        value = inputs_t @ self.Wv

        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

    # ------------------------------------------------------------------ #
    #  Quantum‑graph construction
    # ------------------------------------------------------------------ #
    def build_qnn(self, qnn_arch: Sequence[int] | None = None) -> None:
        """Generate a random quantum‑neural‑network and its training data.

        The helper re‑uses the random‑unitary construction from the second seed
        but keeps the output in a simple list‑of‑lists format that is compatible
        with the original :func:`feedforward` and :func:`state_fidelity`.
        """
        if qnn_arch is None:
            qnn_arch = self.qnn_arch
        # ----- random unitary layers ---------------------------------------
        def _random_unitary(num_qubits: int) -> Tensor:
            dim = 2 ** num_qubits
            mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
            unitary = np.linalg.qr(mat)[0]  # QR decomposition gives unitary
            return torch.from_numpy(np.asarray(unitary, dtype=np.complex128))

        # ----- build layers -------------------------------------------------
        self._qnn_weights = []
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops = []
            for _ in range(num_outputs):
                layer_ops.append(_random_unitary(num_inputs + 1))
            self._qnn_weights.append(layer_ops)

        # ----- training data ------------------------------------------------
        target = self._qnn_weights[-1][0]
        self._qnn_training_data = _random_training_data(target, samples=100)

    def feedforward(self, samples: IterableType[Tuple[Array, Array]]) -> List[List[Tensor]]:
        """Apply the quantum‑feedforward with the same inputs as the quantum
        backend uses in the second seed.  The function returns the
        activations for each layer.  The routine is deliberately lightweight
        and uses pure‑numpy for simplicity.
        """
        if self._qnn_weights is None:
            raise ValueError("call ``build_qnn()`` before feeding forward")
        activations: List[List[Tensor]] = []
        for feature, _ in samples:
            layerwise = [torch.as_tensor(feature, dtype=torch.float32)]
            current = layerwise[0]
            for ops in self._qnn_weights:
                # simple product of all ops in the layer
                state = current
                for gate in ops:
                    state = gate @ state
                layerwise.append(state)
            activations.append(layerwise)
        return activations

    # ------------------------------------------------------------------ #
    #  Fidelity‑based graph construction
    # ------------------------------------------------------------------ #
    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Compute the squared overlap between two quantum states."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm.conj() @ b_norm).abs().item() ** 2)

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from the activations of the quantum feed‑forward.
        The graph is built from the last‑layer activations of each sample
        (the state that *after* all layers).  The node‑wise fidelity
        between each pair of states is computed and edges are added
        if the fidelity exceeds the threshold.
        """
        if self._qnn_weights is None:
            raise ValueError("call ``build_qnn()`` before constructing graph")
        # gather last‑layer states
        last_states = []
        for sample in self._qnn_training_data:
            _, target = sample
            last_states.append(target)
        # compute pairwise fidelities
        graph = nx.Graph()
        graph.add_nodes_from(range(len(last_states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(last_states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Graph utilities from GraphQNN
    # ------------------------------------------------------------------ #
    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = SelfAttention.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def feedforward_static(qnn_arch: Sequence[int],
                           unitaries: Sequence[Sequence[Tensor]],
                           samples: IterableType[Tuple[Array, Array]]) -> List[List[Tensor]]:
        """Static feedforward using pre‑generated unitaries."""
        stored_states: List[List[Tensor]] = []
        for feature, _ in samples:
            activations = [torch.as_tensor(feature, dtype=torch.float32)]
            current = activations[0]
            for layer_ops in unitaries:
                state = current
                for gate in layer_ops:
                    state = gate @ state
                activations.append(state)
            stored_states.append(activations)
        return stored_states

__all__ = ["SelfAttention"]
