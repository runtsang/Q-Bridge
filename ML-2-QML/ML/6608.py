"""Classical implementation of a quanvolutional + graph‑based neural network.

The module defines a single ``QuanvolutionGraphQNN`` class that:
1.  extracts 2×2 patches via a Conv2d layer,
2.  flattens the patches into a dense feature vector,
3.  feeds the vector through a small feed‑forward network,
4.  exposes a method ``fidelity_graph`` that builds a weighted graph
    from cosine similarities of intermediate activations.
"""

import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGraphQNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_channels: int = 4,
        patch_size: int = 2,
        stride: int = 2,
        hidden_dim: int = 128,
        num_classes: int = 10,
        fidelity_threshold: float = 0.8,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        # Classical patch extractor
        self.patch_extractor = nn.Conv2d(
            in_channels, patch_channels, kernel_size=patch_size, stride=stride
        )
        # Feed‑forward head
        self.fc1 = nn.Linear(patch_channels * 14 * 14, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Fidelity graph parameters
        self.fidelity_threshold = fidelity_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_extractor(x)
        flat = patches.view(patches.size(0), -1)
        h = torch.tanh(self.fc1(flat))
        logits = self.fc2(h)
        return F.log_softmax(logits, dim=-1)

    def _activation_sequence(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return intermediate activations for a single sample."""
        patches = self.patch_extractor(x.unsqueeze(0))
        flat = patches.view(1, -1)
        h = torch.tanh(self.fc1(flat))
        return [flat.squeeze(0), h.squeeze(0)]

    def fidelity_graph(self, x_batch: torch.Tensor) -> nx.Graph:
        """Build a graph from cosine similarities of hidden activations."""
        activations = [self._activation_sequence(x) for x in x_batch]
        hidden = torch.stack([a[1] for a in activations])  # (B, hidden_dim)
        graph = nx.Graph()
        graph.add_nodes_from(range(hidden.size(0)))
        for i, j in itertools.combinations(range(hidden.size(0)), 2):
            sim = F.cosine_similarity(hidden[i].unsqueeze(0), hidden[j].unsqueeze(0)).item()
            if sim >= self.fidelity_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary_threshold is not None and sim >= self.secondary_threshold:
                graph.add_edge(i, j, weight=self.secondary_weight)
        return graph
