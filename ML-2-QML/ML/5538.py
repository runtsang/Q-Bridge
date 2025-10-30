"""Hybrid classical‑quantum convolutional network.

The module combines:
* a fast 1×1 convolutional pre‑processor,
* a quantum‑inspired 2×2 patch filter (implemented with torchquantum),
* a fully‑connected autoencoder that learns a latent space,
* a fidelity‑based graph that clusters latent vectors,
* a linear head that maps cluster centroids to class scores.

The design is inspired by the four reference pairs and demonstrates how
classical and quantum ideas can reinforce each other in a single trainable
model.
"""

from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
import networkx as nx
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Classical pre‑processing ------------------------------------------------- #
# --------------------------------------------------------------------------- #
class ConvPreprocessor(nn.Module):
    """Simple 1×1 conv that normalises pixel intensities."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x))


# --------------------------------------------------------------------------- #
# 2. Quantum‑inspired patch filter ------------------------------------------- #
# --------------------------------------------------------------------------- #
class QuanvPatchFilter(tq.QuantumModule):
    """Apply a random two‑qubit circuit to every 2×2 patch of the image."""
    def __init__(self, patch_size: int = 2, n_ops: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = patch_size ** 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )
        self.layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, *_ = x.shape
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        patches = []
        for r in range(0, x.size(-2), self.patch_size):
            for c in range(0, x.size(-1), self.patch_size):
                data = x[..., r:r+self.patch_size, c:c+self.patch_size]
                # flatten 2×2 patch to 4‑dim vector
                data = data.reshape(bsz, -1)
                self.encoder(qdev, data)
                self.layer(qdev)
                out = self.measure(qdev)
                patches.append(out)
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
# 3. Autoencoder ------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder used as a feature extractor."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        decoder.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


# --------------------------------------------------------------------------- #
# 4. Graph clustering -------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two latent vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def build_fidelity_graph(vectors: list[torch.Tensor], threshold: float,
                         secondary: float | None = None,
                         secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(vectors)))
    for (i, vi), (j, vj) in zip(vectors, vectors[1:], strict=False):
        fid = fidelity(vi, vj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

def cluster_centroids(vectors: list[torch.Tensor], G: nx.Graph) -> torch.Tensor:
    """Return the average latent vector per connected component."""
    centroids = []
    for comp in nx.connected_components(G):
        comp_vecs = torch.stack([vectors[i] for i in comp])
        centroids.append(comp_vecs.mean(dim=0))
    return torch.stack(centroids)


# --------------------------------------------------------------------------- #
# 5. Hybrid model ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #
class HybridQuanvolutionNet(nn.Module):
    """Full hybrid network that integrates all building blocks."""
    def __init__(self, num_classes: int = 10, latent_dim: int = 32,
                 fidelity_thr: float = 0.7, secondary_thr: float | None = 0.5):
        super().__init__()
        self.preproc = ConvPreprocessor()
        self.quanv = QuanvPatchFilter()
        self.autoenc = AutoencoderNet(input_dim=28*28, latent_dim=latent_dim)
        self.fidelity_thr = fidelity_thr
        self.secondary_thr = secondary_thr
        # The linear head will receive as many features as clusters
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pre‑process
        x = self.preproc(x)
        # 2. Quantum‑inspired filter
        x = self.quanv(x)
        # 3. Flatten to vectors for autoencoder
        x_flat = x.view(x.size(0), -1)
        # 4. Encode to latent space
        latents = self.autoenc.encode(x_flat)
        # 5. Build fidelity graph
        G = build_fidelity_graph(list(latents), self.fidelity_thr,
                                 self.secondary_thr)
        # 6. Compute cluster centroids
        centroids = cluster_centroids(list(latents), G)
        # 7. Classify
        logits = self.classifier(centroids)
        return logits

__all__ = ["HybridQuanvolutionNet"]
