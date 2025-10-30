"""
Hybrid quantum‑classical model using torchquantum.

The class HybridQuantumNAT is defined as a subclass of tq.QuantumModule.
It mirrors the classical architecture but replaces the variational kernel
with a true quantum circuit built from RandomLayer, general encoders,
and measurement.  The graph regulariser is computed from the measurement
results.  The implementation draws on the QuantumNAT.py and Quanvolution.py seeds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import itertools

# --------------------------------------------------------------------------- #
# 1. Classical feature extractor (same as in ML code)
# --------------------------------------------------------------------------- #
class ConvFeatureExtractor(tq.QuantumModule):
    """
    Classical CNN backbone implemented as a QuantumModule for
    compatibility with torchquantum's graph tracking.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 4) -> None:
        super().__init__()
        self.conv1 = tq.Conv2d(in_ch, 8, kernel_size=3, stride=1, padding=1)
        self.bn1   = tq.BatchNorm2d(8)
        self.pool1 = tq.MaxPool2d(2)

        self.conv2 = tq.Conv2d(8, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2   = tq.BatchNorm2d(out_ch)
        self.pool2 = tq.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        return x

# --------------------------------------------------------------------------- #
# 2. Quantum variational kernel (true quantum circuit)
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """
    Implements a variational circuit that acts on 4‑wire 2×2 patches.
    The circuit consists of a RandomLayer followed by parameterised
    single‑qubit rotations and a small entangling block.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=12, wires=list(range(n_wires)))
        # Parameterised rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Entangling block
        self.cnot = tq.CNOT(has_params=False, trainable=False)
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        # Apply rotations on each wire
        for i in range(self.n_wires):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i)
        # Entangle wires in a simple chain
        for i in range(self.n_wires - 1):
            self.cnot(qdev, wires=[i, i + 1])
        # Measure all qubits
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
# 3. Graph‑based regulariser (classical)
# --------------------------------------------------------------------------- #
class GraphRegulariser(tq.QuantumModule):
    """
    Computes a fidelity‑based graph and Laplacian from the quantum
    measurement outcomes.  The implementation is identical to the
    classical version but wrapped as a QuantumModule for graph
    tracking.
    """
    def __init__(self, threshold: float = 0.8, secondary: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.secondary = secondary

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Normalise
        norm = states / (states.norm(dim=1, keepdim=True) + 1e-12)
        prod = torch.einsum('bi,bj->ij', norm, norm)
        fid  = prod ** 2
        adj = torch.where(fid >= self.threshold, torch.ones_like(fid),
                          torch.where(self.secondary is not None and fid >= self.secondary,
                                      torch.full_like(fid, 0.5), torch.zeros_like(fid)))
        deg = adj.sum(dim=1)
        L = torch.diag(deg) - adj
        return torch.trace(states.t() @ L @ states)

# --------------------------------------------------------------------------- #
# 4. Full hybrid model
# --------------------------------------------------------------------------- #
class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum‑classical hybrid model.  The forward pass follows the same
    high‑level steps as the classical version but uses a true quantum
    circuit for the variational kernel.  The graph regulariser is
    computed from the measurement results.
    """
    def __init__(self,
                 n_classes: int = 10,
                 n_features: int = 4,
                 n_wires: int = 4,
                 graph_thresh: float = 0.75) -> None:
        super().__init__()
        self.extractor   = ConvFeatureExtractor(out_ch=n_features)
        self.quantum     = QuantumKernel(n_wires=n_wires)
        self.regulariser = GraphRegulariser(threshold=graph_thresh)
        # Linear head: flatten the 4‑channel feature map (14×14 = 196)
        self.classifier  = nn.Linear(n_features * 14 * 14, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1. Classical feature extraction
        feat = self.extractor(x)  # (B, 4, 14, 14)

        # 2. Prepare 2×2 patches for the quantum kernel
        patches = feat.unfold(2, 2, 2).unfold(3, 2, 2)
        B, C, H, W, _, _ = patches.shape
        patches = patches.contiguous().view(B * H * W, C, 2, 2)

        # 3. Quantum kernel on each patch
        # Create a quantum device for the batch of patches
        qdev = tq.QuantumDevice(n_wires=self.quantum.n_wires,
                                bsz=B * H * W,
                                device=x.device,
                                record_op=True)
        # Encode each patch into the device
        for i in range(B * H * W):
            # Flatten patch to 4 values
            flat = patches[i].reshape(-1)
            # Simple encoding: rotate each qubit by the pixel value
            for w in range(self.quantum.n_wires):
                self.quantum.rx(qdev, wires=w, params=flat[w])
        # Run the variational circuit
        q_out = self.quantum(qdev)  # (B*H*W, 4)

        # 4. Reshape back to image‑level
        q_out = q_out.view(B, H * W, 4)

        # 5. Flatten for the linear classifier
        flat = q_out.view(B, -1)

        # 6. Graph regulariser term
        reg_term = self.regulariser(q_out.view(B, -1))

        # 7. Linear classification
        logits = self.classifier(flat)

        # 8. Return logits and regularisation scalar
        return logits, reg_term

__all__ = ["HybridQuantumNAT"]
