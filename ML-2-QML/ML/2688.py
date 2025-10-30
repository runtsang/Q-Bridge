import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List

__all__ = ["SamplerQNNGen256"]

class SamplerQNNGen256(nn.Module):
    """
    Hybrid sampler combining a classical MLP and quantum sampling.

    Parameters
    ----------
    qnn_arch : Sequence[int], optional
        Layer sizes for the MLP. Default is (2, 4, 2).
    """
    def __init__(self, qnn_arch: Sequence[int] = (2, 4, 2)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        layers.pop()  # remove last activation
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classical forward pass returning softmax probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, sampler, *args, **kwargs):
        """
        Delegate sampling to an external quantum sampler.

        Parameters
        ----------
        sampler : object
            Must provide a ``sample`` method returning a probability distribution.
        """
        return sampler.sample(*args, **kwargs)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int = 10):
        """Generate a random MLP and a training dataset."""
        weights = [torch.randn(out_f, in_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        target_weight = weights[-1]
        training_data = []
        for _ in range(samples):
            features = torch.randn(qnn_arch[0])
            target = target_weight @ features
            training_data.append((features, target))
        return qnn_arch, weights, training_data, target_weight

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """Compute activations layer‑wise for a set of samples."""
        activations = []
        for features, _ in samples:
            layer_vals = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                layer_vals.append(current)
            activations.append(layer_vals)
        return activations

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph based on pairwise fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = SamplerQNNGen256.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def graph(self, data: Sequence[torch.Tensor], threshold: float,
              *, secondary: float | None = None, secondary_weight: float = 0.5):
        """
        Construct a graph from either logits or state vectors.

        Parameters
        ----------
        data : Sequence[torch.Tensor]
            Either raw logits or quantum state vectors.
        threshold : float
            Fidelity threshold for edge creation.
        """
        return SamplerQNNGen256.fidelity_adjacency(data, threshold,
                                                    secondary=secondary,
                                                    secondary_weight=secondary_weight)
