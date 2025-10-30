"""Quantum implementation of the QuanvolutionGraphQNN architecture.

The class is a subclass of torchquantum.QuantumModule and mirrors
the classical counterpart.  It contains:
1.  A classical Conv2d patch extractor identical to the ML version.
2.  A QuantumPatchMapper that applies a random two‑qubit quantum kernel
    to each 2×2 patch.
3.  A linear classifier on the concatenated quantum feature vector.
4.  Methods to compute a fidelity graph of intermediate quantum states
    using cosine similarity of measurement vectors.
"""

import itertools
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionGraphQNN(tq.QuantumModule):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 2,
        stride: int = 2,
        n_wires: int = 4,
        num_ops: int = 8,
        hidden_dim: int = 128,
        num_classes: int = 10,
        fidelity_threshold: float = 0.8,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        # Classical patch extractor
        self.patch_extractor = nn.Conv2d(
            in_channels, n_wires, kernel_size=patch_size, stride=stride
        )
        self.n_wires = n_wires

        # Quantum patch mapper
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=num_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classifier on concatenated measurement vector
        self.classifier = nn.Linear(n_wires * 14 * 14, num_classes)

        # Fidelity graph parameters
        self.fidelity_threshold = fidelity_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_extractor(x)  # (B, n_wires, 14, 14)
        bsz = patches.size(0)
        device = patches.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        measurement_vectors = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = patches[:, :, r:r + 2, c:c + 2].view(bsz, self.n_wires)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                m = self.measure(qdev)
                measurement_vectors.append(m.view(bsz, self.n_wires))

        features = torch.cat(measurement_vectors, dim=1)  # (B, n_wires*14*14)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def _measurement_sequence(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return concatenated measurement vectors for a single sample."""
        patches = self.patch_extractor(x.unsqueeze(0))
        bsz = patches.size(0)
        device = patches.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        measurement_vectors = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = patches[:, :, r:r + 2, c:c + 2].view(bsz, self.n_wires)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                m = self.measure(qdev)
                measurement_vectors.append(m.view(bsz, self.n_wires))
        return [torch.cat(measurement_vectors, dim=1).squeeze(0)]

    def fidelity_graph(self, x_batch: torch.Tensor) -> nx.Graph:
        """Build a graph from cosine similarities of measurement vectors."""
        vectors = [self._measurement_sequence(x)[0] for x in x_batch]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(vectors)))
        for i, j in itertools.combinations(range(len(vectors)), 2):
            sim = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0)).item()
            if sim >= self.fidelity_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary_threshold is not None and sim >= self.secondary_threshold:
                graph.add_edge(i, j, weight=self.secondary_weight)
        return graph
