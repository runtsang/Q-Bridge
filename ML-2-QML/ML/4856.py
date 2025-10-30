"""Graph‑neural‑network hybrid for classical experiments.

The module implements a ``GraphQNNHybrid`` class that mirrors the
behaviour of the original ``GraphQNN`` seed while adding
estimator/QNN utilities from the EstimatorQNN and
QuantumClassifierModel seeds.  All operations are pure PyTorch,
so the module can be used in any classical training pipeline.

Typical usage::

    from GraphQNN__gen317 import GraphQNNHybrid
    gnn = GraphQNNHybrid(mode="classical")
    arch, weights, data, target = gnn.random_network([3, 5, 2], samples=10)
    activations = gnn.feedforward(arch, weights, data)
    G = gnn.fidelity_adjacency([a[-1] for a in activations],
                                threshold=0.8)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import networkx as nx
import itertools

Tensor = torch.Tensor


class GraphQNNHybrid:
    """Hybrid graph‑neural‑network factory for classical experiments.

    Parameters
    ----------
    mode : {"classical", "quantum"}
        Selects the underlying implementation.  ``"classical"``
        activates a pure PyTorch network; ``"quantum"`` is
        supported by the quantum module.

    Notes
    -----
    The public API is intentionally identical to the original
    ``GraphQNN`` seed and the quantum counterpart, making it
    trivial to swap implementations in downstream code.
    """

    def __init__(self, mode: str = "classical") -> None:
        self.mode = mode
        if mode not in {"classical", "quantum"}:
            raise ValueError("mode must be 'classical' or 'quantum'")

    # ------------------------------------------------------------------ #
    #  Utilities that are common to both modes
    # ------------------------------------------------------------------ #

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the squared cosine similarity between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------ #
    #  Classical‑only helpers
    # ------------------------------------------------------------------ #

    def _random_linear(self, in_f: int, out_f: int) -> Tensor:
        return torch.randn(out_f, in_f, dtype=torch.float32)

    def random_training_data(self, weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Sample training pairs from a linear map."""
        data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(weight.size(1), dtype=torch.float32)
            y = weight @ x
            data.append((x, y))
        return data

    def random_network(self, qnn_arch: Sequence[int], samples: int):
        """Generate a random feed‑forward network with weights."""
        weights: List[Tensor] = [
            self._random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        ]
        target_weight = weights[-1]
        training_data = self.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Propagate inputs through the classical network."""
        activations: List[List[Tensor]] = []
        for x, _ in samples:
            layer_inputs = [x]
            current = x
            for w in weights:
                current = torch.tanh(w @ current)
                layer_inputs.append(current)
            activations.append(layer_inputs)
        return activations

    # ------------------------------------------------------------------ #
    #  Quantum‑only helpers are provided in the quantum module
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"<GraphQNNHybrid mode={self.mode!r}>"


__all__ = ["GraphQNNHybrid"]
