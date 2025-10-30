"""
HybridQuantumNAT: a unified model combining classical CNN, quantum encoder, kernel evaluation, and graph‑based state analysis.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule, RandomLayer, RX, RY, RZ, CRX, MeasureAll, GeneralEncoder, encoder_op_list_name_dict

# --------------------------------------------------------------------------- #
# 1. Classical / Quantum core modules
# --------------------------------------------------------------------------- #

class _QuantumQLayer(QuantumModule):
    """Variational layer used in the hybrid model."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = RandomLayer(n_ops=50, wires=list(range(n_wires)))
        self.rx0 = RX(has_params=True, trainable=True)
        self.ry0 = RY(has_params=True, trainable=True)
        self.rz0 = RZ(has_params=True, trainable=True)
        self.crx0 = CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tq.hadamard(qdev, wires=3)
        tq.sx(qdev, wires=2)
        tq.cnot(qdev, wires=[3, 0])


class QuantumKernel(QuantumModule):
    """Fixed‑parameter quantum kernel based on a simple rotation ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.QuantumModule()
        self.ansatz.forward = self._ansatz_forward

    def _ansatz_forward(self, q_device: QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode two classical vectors into a single circuit."""
        q_device.reset_states(x.shape[0])
        for idx, wire in enumerate(range(self.n_wires)):
            tq.ry(q_device, wires=wire, params=x[:, idx])
        for idx, wire in reversed(list(enumerate(range(self.n_wires)))):
            tq.ry(q_device, wires=wire, params=-y[:, idx])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class _FullyConnectedLayer(nn.Module):
    """Simulated quantum fully‑connected layer with a tunable linear transform."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        vals = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(vals)).mean(dim=0)


# --------------------------------------------------------------------------- #
# 2. Graph utilities (adapted from GraphQNN)
# --------------------------------------------------------------------------- #

def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized dot‑product squared."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_n, b_n).item() ** 2)


def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# 3. Hybrid model
# --------------------------------------------------------------------------- #

class HybridQuantumNAT(nn.Module):
    """
    A hybrid neural architecture that can operate in pure classical mode,
    a quantum‑enhanced mode, or a purely quantum mode.  It exposes a
    `forward` method for inference and helper methods for kernel matrix
    evaluation and graph construction.
    """
    def __init__(self,
                 *,
                 use_quantum: bool = False,
                 use_kernel: bool = False,
                 use_graph: bool = False,
                 num_classes: int = 4,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.use_kernel = use_kernel
        self.use_graph = use_graph
        self.device = torch.device(device)

        if self.use_quantum:
            # Quantum encoder + layer
            self.encoder = GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
            self.q_layer = _QuantumQLayer()
            self.measure = MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(4)
        else:
            # Classical CNN + FC backbone
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )
            self.norm = nn.BatchNorm1d(num_classes)

        if self.use_kernel:
            self.kernel = QuantumKernel()

        if self.use_graph:
            # No state storage here; users call `build_fidelity_graph` externally.
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  For quantum mode the input is expected to be a batch
        of flattened feature vectors (e.g., after pooling).  For classical
        mode the input is a batch of images.
        """
        if self.use_quantum:
            bsz = x.shape[0]
            qdev = QuantumDevice(n_wires=4, bsz=bsz, device=self.device, record_op=True)
            # Encode classical data
            pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
            self.encoder(qdev, pooled)
            # Variational layer
            self.q_layer(qdev)
            # Measurement
            out = self.measure(qdev)
            return self.norm(out)
        else:
            features = self.features(x)
            flattened = features.view(x.shape[0], -1)
            out = self.fc(flattened)
            return self.norm(out)

    # ----------------------------------------------------------------------- #
    # 4. Kernel matrix helper
    # ----------------------------------------------------------------------- #
    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Compute a Gram matrix between two sets of vectors using the
        configured kernel (classical RBF or quantum kernel).
        """
        if not self.use_kernel:
            raise RuntimeError("Kernel functionality is disabled.")
        mat = torch.empty(len(a), len(b))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.kernel(x, y)
        return mat

    # ----------------------------------------------------------------------- #
    # 5. Graph construction helper
    # ----------------------------------------------------------------------- #
    def build_fidelity_graph(self,
                             states: Sequence[torch.Tensor],
                             threshold: float,
                             *,
                             secondary: float | None = None,
                             secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted adjacency graph from a list of state vectors.
        """
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

    # ----------------------------------------------------------------------- #
    # 6. Fully‑connected layer stand‑in
    # ----------------------------------------------------------------------- #
    def fully_connected_layer(self, n_features: int = 1) -> _FullyConnectedLayer:
        """
        Return a lightweight layer that mimics the quantum fully‑connected
        layer used in the original FCL example.
        """
        return _FullyConnectedLayer(n_features)

__all__ = ["HybridQuantumNAT"]
