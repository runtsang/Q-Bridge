"""Hybrid LSTM with a quantum convolutional front‑end for sequence tagging.

The quantum front‑end implements the quanvolution filter using a small
quantum kernel on 2×2 image patches.  Each LSTM gate is a dedicated quantum
circuit that transforms a classical vector into a qubit register, then
measures it to provide the gate activation.  The overall architecture
mirrors the original QLSTM interface while fully exploiting quantum
resources.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumConvFilter(tq.QuantumModule):
    """Applies a random two‑qubit quantum kernel to every 2×2 patch of the input."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QLayer(tq.QuantumModule):
    """Small quantum circuit that can be used as an LSTM gate."""
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
        qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class QLSTM(nn.Module):
    """LSTM cell where all four gates are realized by tiny quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that combines a quantum convolution and a quantum‑gated LSTM."""
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        tagset_size: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.qconv = QuantumConvFilter()
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        conv_features = self.qconv(embeds.unsqueeze(1))
        lstm_out, _ = self.lstm(conv_features.unsqueeze(1))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
