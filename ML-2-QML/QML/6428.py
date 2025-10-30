"""Hybrid quantum model combining QCNN and quantum LSTM for sequence tagging."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

import torchquantum as tq
import torchquantum.functional as tqf

# Quantum LSTM implementation (copy of the seed quantum QLSTM)
class QuantumQLSTM(nn.Module):
    """Quantum LSTM cell where gates are realized by small quantum circuits."""

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

# Quantum QCNN implementation (adapted from the seed QCNN quantum code)
class QuantumQCNN(tq.QuantumModule):
    """Quantum convolutional neural network using torchquantum."""
    def __init__(self) -> None:
        super().__init__()
        # 8‑qubit feature map
        self.feature_map = tq.ZFeatureMap(num_wires=8)
        # Convolutional and pooling layers
        self.conv_layers = [
            self._make_conv_layer(8),
            self._make_conv_layer(4),
            self._make_conv_layer(2),
        ]
        self.pool_layers = [
            self._make_pool_layer([0, 1, 2, 3], [4, 5, 6, 7]),
            self._make_pool_layer([0, 1], [2, 3]),
            self._make_pool_layer([0], [1]),
        ]

    def _conv_circuit(self, params: torch.Tensor) -> tq.QuantumCircuit:
        qc = tq.QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _make_conv_layer(self, num_qubits: int) -> tq.QuantumCircuit:
        qc = tq.QuantumCircuit(num_qubits)
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(self._conv_circuit(torch.rand(3)), [q1, q2])
        return qc

    def _pool_circuit(self, params: torch.Tensor) -> tq.QuantumCircuit:
        qc = tq.QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _make_pool_layer(self, sources, sinks) -> tq.QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = tq.QuantumCircuit(num_qubits)
        for src, sink in zip(sources, sinks):
            qc.append(self._pool_circuit(torch.rand(3)), [src, sink])
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. `x` is (batch, 8)."""
        qdev = tq.QuantumDevice(n_wires=8, bsz=x.shape[0], device=x.device)
        self.feature_map(qdev, x)
        for layer in self.conv_layers + self.pool_layers:
            layer(qdev)
        return tq.MeasureAll(tq.PauliZ)(qdev)

class HybridQLSTMTagger(nn.Module):
    """Hybrid quantum sequence tagging model."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.feature_mapper = nn.Linear(embedding_dim, 8)
        self.qcnn = QuantumQCNN()
        self.qcnn_hidden_mapper = nn.Linear(1, hidden_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        seq_len, batch_size, _ = embeds.shape
        x = self.feature_mapper(embeds)  # (seq_len, batch, 8)
        x = x.reshape(seq_len * batch_size, 8)
        x = self.qcnn(x)  # (seq_len*batch, 8)
        x = x.mean(dim=1, keepdim=True)  # collapse to (seq_len*batch, 1)
        x = self.qcnn_hidden_mapper(x)  # (seq_len*batch, hidden_dim)
        x = x.reshape(seq_len, batch_size, self.hidden_dim)
        lstm_out, _ = self.lstm(x)  # (seq_len, batch, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out.view(seq_len, -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTMTagger"]
