"""Hybrid classical‑quantum architecture that integrates Conv, QFCModel, and EstimatorQNN.

The module is composed of:
* A classical Conv filter that turns each image into a scalar feature.
* A quantum feature extractor (QFCModel) based on torchquantum.
* A classical regression head (EstimatorQNN).
* A helper that builds a weighted adjacency graph from pairwise fidelities of quantum outputs.

Each component is fully modular, allowing independent upgrades while preserving the overall pipeline.
"""

import torch
import torch.nn as nn
import networkx as nx
from Conv import Conv
from QuantumNAT import QFCModel
from EstimatorQNN import EstimatorQNN
from GraphQNN import state_fidelity


class HybridNAT(nn.Module):
    """Hybrid classical‑quantum model inspired by Quantum‑NAT, GraphQNN and EstimatorQNN."""

    def __init__(self):
        super().__init__()
        # Classical pre‑processing
        self.conv = Conv()(kernel_size=2, threshold=0.0)
        # Quantum feature extractor
        self.quantum = QFCModel()
        # Classical regression head
        self.regressor = EstimatorQNN()
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalized scalar predictions of shape (B, 1).
        """
        # 1. Classical convolution filter applied to each image
        conv_features = []
        for img in x:
            # img is (1, H, W) -> (H, W)
            conv_val = self.conv.run(img.squeeze(0).cpu().numpy())
            conv_features.append(conv_val)
        conv_tensor = torch.tensor(conv_features, dtype=torch.float32, device=x.device).unsqueeze(1)

        # 2. Quantum feature extraction
        quantum_out = self.quantum(conv_tensor)

        # 3. Classical regression
        out = self.regressor(quantum_out)

        return self.norm(out)

    def graph_from_outputs(self, outputs: torch.Tensor) -> nx.Graph:
        """
        Build a weighted fidelity graph from a batch of quantum outputs.

        Parameters
        ----------
        outputs : torch.Tensor
            Tensor of shape (B, 1) produced by the quantum module.

        Returns
        -------
        networkx.Graph
            Weighted adjacency graph where edges represent high fidelity between
            quantum outputs of different samples.
        """
        vecs = outputs.squeeze(1).cpu().numpy()
        fidelities = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                fid = state_fidelity(torch.tensor(vecs[i]), torch.tensor(vecs[j]))
                fidelities.append((i, j, fid))

        G = nx.Graph()
        G.add_nodes_from(range(len(vecs)))
        for i, j, fid in fidelities:
            if fid >= 0.9:
                G.add_edge(i, j, weight=1.0)
            elif fid >= 0.7:
                G.add_edge(i, j, weight=0.5)
        return G


__all__ = ["HybridNAT"]
