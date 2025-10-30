"""
HybridQuantumNAT: quantum‑centric implementation using TorchQuantum.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule, RandomLayer, RX, RY, RZ, CRX, MeasureAll, GeneralEncoder, encoder_op_list_name_dict

# --------------------------------------------------------------------------- #
# 1. Quantum core modules
# --------------------------------------------------------------------------- #

class _QuantumQLayer(QuantumModule):
    """Variational layer for the hybrid model."""
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
    """Quantum kernel with a fixed rotation ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.QuantumModule()
        self.ansatz.forward = self._ansatz_forward

    def _ansatz_forward(self, q_device: QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
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


# --------------------------------------------------------------------------- #
# 2. Graph utilities
# --------------------------------------------------------------------------- #

def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_n, b_n).item() ** 2)


def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
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
# 3. Hybrid quantum model
# --------------------------------------------------------------------------- #

class HybridQuantumNAT(QuantumModule):
    """
    Quantum‑centric hybrid model that can operate purely quantumly or
    fallback to a classical CNN backbone via a `use_classical` flag.
    """
    def __init__(self,
                 *,
                 use_classical: bool = False,
                 use_kernel: bool = False,
                 use_graph: bool = False,
                 num_classes: int = 4,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.use_classical = use_classical
        self.use_kernel = use_kernel
        self.use_graph = use_graph
        self.device = torch.device(device)

        if self.use_classical:
            # Classical CNN backbone (executed on CPU)
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
        else:
            # Quantum encoder + variational layer
            self.encoder = GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
            self.q_layer = _QuantumQLayer()
            self.measure = MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(4)

        if self.use_kernel:
            self.kernel = QuantumKernel()

        if self.use_graph:
            # Graph utilities are exposed via a helper method
            pass

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_classical:
            features = self.features(x)
            flattened = features.view(x.shape[0], -1)
            out = self.fc(flattened)
            return self.norm(out)
        else:
            bsz = x.shape[0]
            qdev = QuantumDevice(n_wires=4, bsz=bsz, device=self.device, record_op=True)
            pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)

    # ----------------------------------------------------------------------- #
    # 4. Kernel matrix helper
    # ----------------------------------------------------------------------- #
    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> torch.Tensor:
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
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

__all__ = ["HybridQuantumNAT"]
