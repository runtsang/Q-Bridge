import torch
import torch.nn as nn
import torchquantum as tq
import networkx as nx
import itertools
import numpy as np
import qutip as qt

class QuantumSelfAttention(tq.QuantumModule):
    """Parameterised quantum self‑attention block."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.rotation = tq.RandomLayer(n_ops=2*n_qubits, wires=range(n_qubits))
        self.crot = tq.CRX(wires=list(range(n_qubits-1)), params=np.zeros(n_qubits-1))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.rotation(qdev)
        for i in range(self.n_qubits-1):
            self.crot(qdev, wires=(i, i+1))
        return self.measure(qdev)

class QLayer(tq.QuantumModule):
    """Simplified Quantum‑NAT QLayer."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random = tq.RandomLayer(n_ops=30, wires=range(self.n_wires))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        return

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum analogue of a classical convolution."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.entangle = tq.RandomLayer(n_ops=6, wires=range(self.n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, features: torch.Tensor):
        # features: (batch, 4)
        for f in features.split(1, dim=1):
            self.encoder(qdev, f.squeeze(-1))
            self.entangle(qdev)
        return self.measure(qdev)

class HybridQuanvolutionQuantumClassifier(tq.QuantumModule):
    """End‑to‑end quantum model mirroring the classical HybridQuanvolutionClassifier."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.n_wires = 4
        self.qfilter = QuanvolutionFilter()
        self.qlayer = QLayer()
        self.attn = QuantumSelfAttention(n_qubits=self.n_wires)
        self.norm = nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Extract first four pixels of each image as a simple feature vector
        feat = x.view(bsz, -1)[:, :4]
        self.qfilter(qdev, feat)

        self.qlayer(qdev)
        attn_out = self.attn(qdev)

        out = self.norm(attn_out)
        return out

def fidelity_adjacency(
    states: list[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5
) -> nx.Graph:
    """Construct a weighted graph from fidelities of pure qutip states."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = abs((a.dag() * b)[0, 0]) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "QuantumSelfAttention",
    "QLayer",
    "QuanvolutionFilter",
    "HybridQuanvolutionQuantumClassifier",
    "fidelity_adjacency",
]
