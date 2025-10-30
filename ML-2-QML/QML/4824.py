"""Quantum‑enhanced LSTM with QCNN feature extraction for sequence tagging."""

from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

# --------------------------------------------------------------------------- #
#  QCNN feature map – quantum convolution + pooling
# --------------------------------------------------------------------------- #

class QCNNFeatureMap(tq.QuantumModule):
    """QCNN that acts as a feature extractor before the LSTM."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_map = tq.ZFeatureMap(n_qubits)
        self.convs = nn.ModuleList([tq.Conv2D(n_qubits) for _ in range(2)])
        self.pools = nn.ModuleList([tq.Pool2D(n_qubits) for _ in range(2)])
        self.final = tq.RX(n_qubits, trainable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.feature_map(dev, x)
        for conv in self.convs:
            conv(dev)
        for pool in self.pools:
            pool(dev)
        self.final(dev)
        return tq.MeasureAll(tq.PauliZ)(dev)


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell
# --------------------------------------------------------------------------- #

class QLSTM(tq.QuantumModule):
    """Variational quantum LSTM where each gate is a small quantum circuit."""

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(dev, x)
            for wire, gate in enumerate(self.params):
                gate(dev, wires=wire)
            return self.measure(dev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#  Hybrid quantum tagger
# --------------------------------------------------------------------------- #

class HybridQLSTMTagger(tq.QuantumModule):
    """Sequence tagger that first projects data through a QCNN feature map
    and then processes it with a variational quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.qcnn_feat = QCNNFeatureMap(n_qubits)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.classifier = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(sentence)
        # QCNN feature extraction
        feats = self.qcnn_feat(embeds)
        lstm_out, _ = self.lstm(feats.unsqueeze(1))
        logits = self.classifier(lstm_out.squeeze(1))
        return torch.log_softmax(logits, dim=1)


__all__ = [
    "HybridQLSTMTagger",
    "QLSTM",
    "QCNNFeatureMap",
]
