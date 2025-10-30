"""Hybrid classical model that fuses quanvolution, kernel and graph pruning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

Tensor = torch.Tensor


class StochasticQuanvolutionFilter(nn.Module):
    """
    A drop‑in replacement for the classical conv2d used in the original
    quanvolution example.  The filter first partitions the image into
    2×2 patches and runs a *parameter‑free* quantum kernel
    (``QuantumKernel`` below) on each patch.  The result is a scalar
    that is fed into a 1‑D convolution so that the filter can be
    trained end‑to‑end.  The design is inspired by the
    classical “Conv” from reference [2] and the quantum kernel
    implementation in reference [3].
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.kernel = QuantumKernel()
        # 1‑D conv to fuse the output scalars from each patch
        self.conv1d = nn.Conv1d(1, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tensor:
        # x: (B, 1, H, W) – MNIST style
        bsz, _, h, w = x.shape
        assert h % self.kernel_size == 0 and w % self.kernel_size == 0, (
            "Image dimensions must be divisible by kernel_size."
        )
        patches = x.unfold(2, self.kernel_size, self.kernel_size) \
                   .unfold(3, self.kernel_size, self.kernel_size)  # (B,1,Nh,Nw,ks,ks)
        patches = patches.contiguous().view(bsz, -1, self.kernel_size, self.kernel_size)
        # Run the quantum kernel on each patch
        results = torch.stack([self.kernel(patch) for patch in patches], dim=1)  # (B, Np, 1)
        # Fuse with a 1‑D conv
        out = self.conv1d(results.transpose(1, 2))  # (B,1,Np)
        return out.squeeze(1)  # (B,Np)


class QuantumKernel(nn.Module):
    """
    Simple parameter‑free quantum kernel inspired by the
    TorchQuantum ``KernalAnsatz`` from reference [3].  It encodes
    each pixel of a 2×2 patch into a single qubit and measures
    the overlap with a fixed reference state |0⟩⟨0|.  The output
    is a scalar in [0,1] that describes how close the patch is to the
    reference state.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_qubits = 4
        # fixed reference state |0...0>
        self.ref_state = torch.zeros(self.n_qubits)
        # no learnable parameters – the kernel is deterministic

    def forward(self, patch: Tensor) -> Tensor:
        # patch: (B, k, k)
        patch = patch.view(patch.shape[0], -1)  # (B,4)
        # encode each pixel as a rotation around Y
        # (B,4,1) -> (B,4)
        enc = torch.atan2(patch[:, 1], patch[:, 0])  # simple heuristic
        # compute overlap with |0⟩, ignoring bias
        overlap = torch.exp(-torch.sum((patch - self.ref_state) ** 2, dim=1))
        return overlap.unsqueeze(1)  # (B,1)


class KernelMixingLayer(nn.Module):
    """
    Mixes the feature maps produced by the stochastic quanvolution
    filter with a classical RBF kernel (see reference [3]) to
    produce a higher‑level representation.  The RBF kernel
    is applied pairwise between the feature vectors of adjacent
    *batch* samples, and the resulting Gram matrix is back‑propagated
    through a learned weight matrix.
    """
    def __init__(self, in_features: int, out_features: int, gamma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.rbf = nn.Linear(in_features, in_features)  # placeholder for RBF
        self.out = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F)
        B, F = x.shape
        # compute Gram matrix (B,B)
        gram = torch.cdist(x, x, p=2)  # (B,B)
        gram = torch.exp(-self.rbf.weight.abs() * gram)  # (B,B)
        # project back to feature space
        return self.out(gram @ x)  # (B, out_features)


class GraphPruner(nn.Module):
    """
    Uses the graph‑based fidelity adjacency from reference [4]
    to create a weight mask that zeros out connections below a
    threshold.  The mask is applied to the final linear classifier
    to keep only the states that are useful for discrimination.
    """
    def __init__(self, weight: Tensor, threshold: float = 0.9, secondary: float | None = None):
        super().__init__()
        self.register_buffer('mask', self._build_mask(weight, threshold, secondary))

    def _build_mask(self, weight: Tensor, threshold: float):
        # weight: (out_features, in_features)
        flat = weight.reshape(-1)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(flat)))
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                fid = torch.dot(weight[i], weight[j]).abs() ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        # mask only the connection if any edge exists
        mask = torch.zeros_like(weight)
        for i, j in graph.edges():
            mask[i, j] = 1
        return mask

    def forward(self, x: Tensor) -> Tensor:
        return x * self.mask


class QuanvolutionFusion(nn.Module):
    """
    End‑to‑end model that stitches together the stochastic quantum filter,
    a mixing kernel layer, and a graph‑based pruning mask.
    """
    def __init__(self, num_classes: int = 10, kernel_size: int = 2) -> None:
        super().__init__()
        self.qfilter = StochasticQuanvolutionFilter(kernel_size=kernel_size)
        # The output dimension after filtering is (B, Np)
        # where Np = (H/kernel_size) * (W/kernel_size)
        # we will flatten for the classifier
        self.fc = nn.Linear(self._calc_flattened_dim(), num_classes)
        self.pruner = GraphPruner(self.fc.weight, threshold=0.95)

    def _calc_flattened_dim(self) -> int:
        dummy = torch.zeros(1, 1, 28, 28)
        out = self.qfilter(dummy)
        return out.shape[1]  # (B,Np)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.qfilter(x)
        x = self.pruner(x)
        # flatten after pruning
        x = x.view(x.shape[0], -1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFusion"]
