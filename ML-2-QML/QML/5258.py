"""QuantumHybridNet – quantum‑enhanced hybrid architecture.

This module implements a hybrid neural network that fuses:
- a CNN backbone (identical to the classical version)
- an optional quantum LSTM (QLSTM from the quantum seed)
- a variational quantum feature extractor
- a regression head
- graph‑based feature aggregation using qutip fidelities

The API matches the classical counterpart; only the internal
implementations differ.  The class can be used as a drop‑in
replacement for the classical version.

Author: gpt-oss-20b
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import qutip as qt
import scipy as sc
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# 1. CNN backbone – identical to classical version
# --------------------------------------------------------------------------- #
class _CNNBase(tq.QuantumModule):
    def __init__(self, in_channels: int = 1, out_features: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 2. Quantum feature extractor – variational circuit
# --------------------------------------------------------------------------- #
class _QuantumFeatureLayer(tq.QuantumModule):
    """Variational circuit that maps 4 classical features to a quantum state."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_qubits)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_qubits):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

# --------------------------------------------------------------------------- #
# 3. Quantum LSTM – adapted from the quantum seed
# --------------------------------------------------------------------------- #
class QuantumLSTM(tq.QuantumModule):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# 4. Graph‑based feature aggregation using qutip fidelity
# --------------------------------------------------------------------------- #
def _qutip_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap between pure states a and b."""
    return abs((a.dag() * b)[0, 0]) ** 2

def _fidelity_graph_qutip(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, ai in enumerate(states):
        for j in range(i + 1, len(states)):
            bj = states[j]
            fid = _qutip_fidelity(ai, bj)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 5. Hybrid network – quantum‑enhanced
# --------------------------------------------------------------------------- #
class QuantumHybridNet(tq.QuantumModule):
    """
    Quantum‑enhanced hybrid model that fuses a CNN backbone,
    an optional quantum LSTM, a variational quantum layer,
    and a regression head.  The API mirrors the classical
    counterpart for seamless interchangeability.
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_qubits: int = 4,
        lstm_n_qubits: int = 0,
        hidden_dim: int = 32,
        output_dim: int = 1,
        graph_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self.cnn = _CNNBase(in_channels, out_features=4)
        self.lstm_n_qubits = lstm_n_qubits
        if lstm_n_qubits > 0:
            self.lstm = QuantumLSTM(4, hidden_dim, n_qubits=lstm_n_qubits)
        else:
            self.lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        self.quantum_layer = _QuantumFeatureLayer(n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.regression_head = nn.Sequential(
            nn.Linear(n_qubits + hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )
        self.graph_threshold = graph_threshold
        # Linear embedding to match feature dimensionality
        self.embed = nn.Linear(4, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image batch.
        """
        cnn_feat = self.cnn(x)          # shape (B, 4)
        if isinstance(self.lstm, QuantumLSTM):
            lstm_out, _ = self.lstm(cnn_feat.unsqueeze(1))
            lstm_feat = lstm_out.squeeze(1)  # shape (B, hidden_dim)
        else:
            lstm_out, _ = self.lstm(cnn_feat.unsqueeze(1))
            lstm_feat = lstm_out.squeeze(1)
        # Quantum feature extraction
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.quantum_layer.n_qubits, bsz=bsz, device=x.device)
        embed = self.embed(cnn_feat)  # shape (B, n_qubits)
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{self.quantum_layer.n_qubits}xRy"])
        encoder(qdev, embed)
        self.quantum_layer(qdev)
        qfeat = self.measure(qdev)  # shape (B, n_qubits)
        combined = torch.cat([lstm_feat, qfeat], dim=1)
        out = self.regression_head(combined)
        return out

    def build_fidelity_graph(self, qstates: Sequence[qt.Qobj], threshold: float | None = None) -> nx.Graph:
        """
        Build a graph from a list of pure quantum states using fidelity.
        """
        thresh = threshold if threshold is not None else self.graph_threshold
        return _fidelity_graph_qutip(qstates, thresh)

__all__ = ["QuantumHybridNet"]
