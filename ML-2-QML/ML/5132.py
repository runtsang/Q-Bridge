import torch
from torch import nn
import torch.nn.functional as F
import itertools
import networkx as nx
import numpy as np

class HybridConvNet(nn.Module):
    """
    Classical hybrid convolutional network that combines a 2‑D convolution,
    a simple estimator, and graph‑based fidelity utilities.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 estimator: nn.Module | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.estimator = estimator

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Run the convolutional filter and optionally pass the mean activation
        through the estimator.

        Parameters
        ----------
        data : torch.Tensor
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            If an estimator is supplied, the regression output is returned.
            Otherwise the mean activation is returned.
        """
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        mean_val = activations.mean()
        if self.estimator is not None:
            inp = torch.tensor([mean_val.item(), 1.0], device=data.device)
            return self.estimator(inp)
        return mean_val

    # ------------------------------------------------------------------
    # Graph‑based utilities (adapted from GraphQNN.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int):
        dataset = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: list[int], samples: int):
        weights = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(HybridConvNet._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = HybridConvNet.random_training_data(target_weight, samples)
        return qnn_arch, weights, training_data, target_weight

    @staticmethod
    def feedforward(qnn_arch: list[int], weights: list[torch.Tensor],
                    samples: list[tuple[torch.Tensor, torch.Tensor]]):
        stored = []
        for features, _ in samples:
            activations = [features]
            current = features
            for w in weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: list[torch.Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridConvNet.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

# ------------------------------------------------------------------
# Helper neural networks
# ------------------------------------------------------------------
class SamplerNet(nn.Module):
    """Simple feed‑forward sampler that outputs a probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class EstimatorNN(nn.Module):
    """Simple regression network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

__all__ = ["HybridConvNet", "SamplerNet", "EstimatorNN"]
