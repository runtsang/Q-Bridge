import torch
import torch.nn as nn
import itertools
import networkx as nx
from typing import Iterable, Sequence, List, Tuple

Tensor = torch.Tensor


class GraphQNN__gen292(nn.Module):
    """
    Classical graph neural network with a learnable feature‑mixing layer
    and optional Gaussian noise injection.  The architecture mirrors the
    original seed but adds a *mix_dim* that can remix node features
    before the standard tanh activations.  The class also exposes a
    convenience ``random_network`` helper that returns a synthetic
    architecture, weights, training data and a target weight.
    """

    def __init__(self, arch: Sequence[int], mix_dim: int | None = None):
        super().__init__()
        self.arch = list(arch)
        self.mix_dim = mix_dim or self.arch[0]
        self.mix_linear = nn.Linear(self.arch[0], self.mix_dim, bias=False)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, features: torch.Tensor) -> Tuple[List[Tensor], Tensor]:
        """
        Forward pass that returns per‑layer activations and the final output.

        Parameters
        ----------
        features : torch.Tensor
            Shape (N_nodes, in_features) – raw node features.

        Returns
        -------
        activations : list[Tensor]
            Activations after each linear layer.
        final_out : Tensor
            Final output of the network.
        """
        mixed = self.mix_linear(features)
        mixed = mixed + torch.randn_like(mixed) * 0.05  # controlled noise
        activations = [mixed]
        current = mixed
        for layer in self.layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        return activations, current

    # ------------------------------------------------------------------
    #  Utility helpers matching the original seed
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(seed: int | None = None) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Generate a synthetic network with random weights and a training set
        that targets the last layer weight matrix.

        Returns
        -------
        arch : list[int]
            Layer sizes.
        weights : list[Tensor]
            Weight matrices for each linear layer.
        training_data : list[tuple[Tensor, Tensor]]
            Features and targets for training.
        target_weight : Tensor
            The last weight matrix that the training data is based on.
        """
        torch.manual_seed(seed)
        arch = [5, 12, 9, 4]
        weights = [torch.randn(o, i) for i, o in zip(arch[:-1], arch[1:])]
        target_weight = weights[-1]
        training_data = GraphQNN__gen292.random_training_data(target_weight, samples=200)
        return arch, weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """
        Create a training set where each target is the linear transformation
        of a random feature vector by the supplied weight matrix.

        Parameters
        ----------
        weight : torch.Tensor
            Target weight matrix.
        samples : int
            Number of training examples.

        Returns
        -------
        dataset : list[tuple[Tensor, Tensor]]
            (features, target) pairs.
        """
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """
        Replicate the original ``feedforward`` but return activations from
        the mixed‑feature network.

        Parameters
        ----------
        samples : iterable of (features, target) tuples

        Returns
        -------
        outputs : list[list[Tensor]]
            Layerwise activations for each sample.
        """
        outputs: List[List[Tensor]] = []
        for features, _ in samples:
            act, _ = self.forward(features)
            outputs.append(act)
        return outputs

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """
        Compute the squared overlap between two output vectors.

        Parameters
        ----------
        a, b : torch.Tensor
            Output vectors.

        Returns
        -------
        fidelity : float
            Squared overlap.
        """
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).pow(2).item())

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from state fidelities.

        Parameters
        ----------
        states : sequence of torch.Tensor
            Output vectors.
        threshold : float
            Primary fidelity threshold.
        secondary : float | None
            Secondary threshold for weaker edges.
        secondary_weight : float
            Weight assigned to secondary edges.

        Returns
        -------
        graph : networkx.Graph
            Weighted adjacency graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "GraphQNN__gen292",
]
