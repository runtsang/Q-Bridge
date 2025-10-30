"""
QuantumHybridNet: a hybrid classical‑quantum model that merges concepts
from the four reference pairs.

The model consists of:
    * A lightweight CNN encoder (QuantumNAT + QCNN).
    * A patch‑wise quantum filter (Quanvolution + QuantumNAT QLayer).
    * A graph‑based regulariser (GraphQNN).
    * A linear classifier.

The quantum filter is implemented with torchquantum and can be
plugged into the classical pipeline seamlessly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
from typing import List, Tuple

# 1. Classical encoder
class _CNNEncoder(nn.Module):
    """CNN encoder inspired by QuantumNAT and QCNN."""
    def __init__(self, in_channels: int = 1, intermediate: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 28 → 14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 14 → 7
        )
        self.feature_map = nn.Linear(16 * 7 * 7, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.feature_map(x))

# 2. Quantum patch filter
class QuantumPatchFilter(tq.QuantumModule):
    """Variational circuit that operates on 2×2 image patches."""
    def __init__(self, n_wires: int = 4, n_random_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_random_ops, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    def encoder(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> None:
        """Encode 4 pixel values into single‑qubit rotations."""
        for w in range(self.n_wires):
            tq.RY(data[:, w], wires=w)(qdev)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])

    def forward_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (batch, num_patches, 4) – flattened 2×2 patches.
        Returns: (batch, num_patches, n_wires) – measurement results.
        """
        bsz, num_patches, _ = patches.shape
        device = patches.device
        out = torch.empty(bsz, num_patches, self.n_wires, device=device)
        for i in range(num_patches):
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=bsz,
                device=device,
                record_op=True,
            )
            self.encoder(qdev, patches[:, i, :])
            self.forward(qdev)
            out[:, i, :] = tqf.measure_all(qdev, wires=range(self.n_wires))
        return out.view(bsz, -1)

# 3. Graph‑based regulator
class GraphRegulator(nn.Module):
    """Builds a fidelity‑based Laplacian from quantum outputs."""
    def __init__(self, threshold: float = 0.95, secondary: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.secondary = secondary

    def forward(self, quantum_states: torch.Tensor) -> torch.Tensor:
        """
        quantum_states: (batch, num_states, n_wires)
        Output: (batch, num_states, num_states) – Laplacian matrices.
        """
        batch, num_states, _ = quantum_states.shape
        lap = torch.empty(batch, num_states, num_states, device=quantum_states.device)
        for b in range(batch):
            states = quantum_states[b]  # (num_states, n_wires)
            G = nx.Graph()
            G.add_nodes_from(range(num_states))
            for i in range(num_states):
                for j in range(i + 1, num_states):
                    fid = torch.dot(states[i], states[j]) ** 2
                    if fid >= self.threshold:
                        G.add_edge(i, j, weight=1.0)
                    elif self.secondary is not None and fid >= self.secondary:
                        G.add_edge(i, j, weight=self.secondary)
            L = torch.zeros(num_states, num_states, device=quantum_states.device)
            for i, j, w in G.edges(data="weight"):
                L[i, j] = -w
                L[j, i] = -w
                L[i, i] += w
                L[j, j] += w
            lap[b] = L
        return lap

# 4. Full hybrid network
class QuantumHybridNet(nn.Module):
    """End‑to‑end hybrid classical‑quantum model."""
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 n_qwires: int = 4,
                 graph_threshold: float = 0.95,
                 graph_secondary: float | None = None) -> None:
        super().__init__()
        self.encoder = _CNNEncoder(in_channels)
        self.qfilter = QuantumPatchFilter(n_wires=n_qwires)
        self.head = nn.Linear(9 * n_qwires, num_classes)
        self.regulator = GraphRegulator(graph_threshold, graph_secondary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoder
        encoded = self.encoder(x)

        # Extract 2×2 patches from the raw image
        patches = F.unfold(
            x, kernel_size=2, stride=2
        )  # (batch, 4, num_patches)
        patches = patches.permute(0, 2, 1)  # (batch, num_patches, 4)

        # Quantum filter
        qout = self.qfilter.forward_patches(patches)  # (batch, num_patches, n_wires)

        # Optional graph regularisation (provides Laplacian)
        _ = self.regulator(qout)

        # Flatten and classify
        flat = qout.view(x.size(0), -1)
        logits = self.head(flat)
        return logits

__all__ = ["QuantumHybridNet"]
