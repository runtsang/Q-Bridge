import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import List, Sequence, Tuple, Iterable

# ----------------------------------------------------------------------
# 1. Classical convolutional primitives (inspired by Conv.py & Quanvolution.py)
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """Simple 2×2 convolution filter with a sigmoid activation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class QuanvolutionFilter(nn.Module):
    """Stride‑2 2×2 convolution that emulates the quantum filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


# ----------------------------------------------------------------------
# 2. Core GraphQNN class (mirrors GraphQNN.py but enriched)
# ----------------------------------------------------------------------
class GraphQNNEnhanced:
    """
    Hybrid Graph Neural Network that can operate in a classical mode
    (PyTorch + RBF kernel) or a quantum mode (Qiskit + variational kernel).
    """

    def __init__(self,
                 arch: Sequence[int],
                 kernel_type: str = "rbf",
                 conv_type: str = "conv",
                 device: str = "cpu"):
        self.arch = list(arch)
        self.kernel_type = kernel_type
        self.conv_type = conv_type
        self.device = device
        self._build_layers()

    def _build_layers(self) -> None:
        # Choose convolution wrapper
        if self.conv_type == "conv":
            self.conv = ConvFilter()
        elif self.conv_type == "quanv":
            self.conv = QuanvolutionFilter()
        else:
            self.conv = None

        # Linear layers
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.layers.to(self.device)

    # ------------------------------------------------------------------
    # 3. Random network generation (weights + synthetic data)
    # ------------------------------------------------------------------
    def random_network(self, samples: int = 100) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        self.weights = [torch.randn(out, in_, dtype=torch.float32) for in_, out in zip(self.arch[:-1], self.arch[1:])]
        target_weight = self.weights[-1]
        data = torch.randn(samples, self.arch[0], dtype=torch.float32)
        targets = data @ target_weight.t()
        return self.arch, self.weights, list(zip(data, targets)), target_weight

    # ------------------------------------------------------------------
    # 4. Forward propagation (captures intermediate states)
    # ------------------------------------------------------------------
    def feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        states: List[List[torch.Tensor]] = []
        for x, _ in samples:
            layerwise: List[torch.Tensor] = [x]
            h = x
            if self.conv is not None:
                h = torch.tensor(self.conv.run(h.cpu().numpy()), dtype=torch.float32, device=self.device)
                layerwise.append(h)
            for layer in self.layers:
                h = torch.tanh(layer(h))
                layerwise.append(h)
            states.append(layerwise)
        return states

    # ------------------------------------------------------------------
    # 5. Fidelity utilities
    # ------------------------------------------------------------------
    def state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency(self,
                           states: Sequence[torch.Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1:], start=i + 1):
                fid = self.state_fidelity(a, b)
                if fid >= threshold:
                    G.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    # 6. Kernel matrix (classical RBF or quantum placeholder)
    # ------------------------------------------------------------------
    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        if self.kernel_type == "rbf":
            gamma = 1.0
            return np.array([[torch.exp(-gamma * torch.sum((x - y) ** 2)).item()
                              for y in b] for x in a])
        else:
            # quantum kernel will be implemented in the QML side
            return np.zeros((len(a), len(b)))

    def __repr__(self) -> str:
        return f"<GraphQNNEnhanced arch={self.arch} kernel={self.kernel_type} conv={self.conv_type}>"

__all__ = ["GraphQNNEnhanced", "ConvFilter", "QuanvolutionFilter"]
