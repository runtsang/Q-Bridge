import torch
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor

class GraphQNN:
    """
    Classical graph‑based neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[3, 5, 2]``.
    residual : bool, default=True
        If ``True`` each layer receives a residual connection from its input.
    """

    def __init__(self, arch: Sequence[int], residual: bool = True):
        self.arch = list(arch)
        self.residual = residual
        self.weights: List[Tensor] = [
            torch.randn(out, in_, dtype=torch.float32, requires_grad=True)
            for in_, out in zip(self.arch[:-1], self.arch[1:])
        ]

    # ------------------------------------------------------------------
    #  Utility constructors
    # ------------------------------------------------------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int, batch_size: int = 1):
        """
        Generate a random network together with a training dataset.

        Returns
        -------
        arch, weights, training_data, target_weight
        """
        weights = [
            GraphQNN._random_linear(in_, out)
            for in_, out in zip(arch[:-1], arch[1:])
        ]
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples, batch_size)
        return arch, weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int,
                             batch_size: int = 1) -> List[Tuple[Tensor, Tensor]]:
        """
        Generate ``samples`` pairs of (input, target) with optional batching.
        """
        dataset: List[Tuple[Tensor, Tensor]] = []
        in_features = weight.size(1)
        out_features = weight.size(0)
        for _ in range(samples):
            inputs = torch.randn(batch_size, in_features, dtype=torch.float32)
            targets = (weight @ inputs.T).T
            dataset.append((inputs, targets))
        return dataset

    # ------------------------------------------------------------------
    #  Forward propagation
    # ------------------------------------------------------------------
    def feedforward(self,
                    samples: Iterable[Tuple[Tensor, Tensor]]
                   ) -> List[List[Tensor]]:
        """
        Run a batch of inputs through the network.

        Returns a list of activations per sample; each activation list contains
        the input followed by the output of every layer.
        """
        stored: List[List[Tensor]] = []
        for inputs, _ in samples:
            activations: List[Tensor] = [inputs]
            current = inputs
            for weight in self.weights:
                current = torch.tanh(weight @ current.T).T
                if self.residual:
                    current = current + activations[-1]
                activations.append(current)
            stored.append(activations)
        return stored

    # ------------------------------------------------------------------
    #  Fidelity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """
        Return the squared overlap between two (possibly batched) state vectors.
        """
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-12)
        b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-12)
        overlap = torch.sum(a_norm * b_norm, dim=-1).abs()
        return float(overlap.pow(2).mean().item())

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted graph where edges represent fidelity between states.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Loss based on graph adjacency
    # ------------------------------------------------------------------
    def fidelity_loss(self,
                      outputs: List[Tensor],
                      graph: nx.Graph) -> torch.Tensor:
        """
        Compute a graph‑weighted loss: sum (1 - fidelity) over all edges.
        """
        loss = 0.0
        for i, j, data in graph.edges(data=True):
            fi = outputs[i]
            fj = outputs[j]
            fid = self.state_fidelity(fi, fj)
            loss += (1.0 - fid) * data.get('weight', 1.0)
        return torch.tensor(loss / graph.number_of_edges(), dtype=torch.float32)

    __all__ = [
        "GraphQNN",
    ]
