"""QuantumHybridNAT: Quantum‑enhanced implementation for image‑to‑sequence tasks.

This module builds on the classical version by adding a quantum projection
for the image encoder and a quantum‑enhanced LSTM for the sequence encoder.
Both quantum components are implemented with TorchQuantum and can be switched
off by setting the corresponding qubit counts to zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["QuantumHybridNAT"]


class _CNNBackbone(nn.Module):
    """CNN backbone identical to the original Quantum‑NAT model."""

    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _QuantumImageEncoder(nn.Module):
    """CNN + quantum projection that mixes the 4‑dim classical feature vector."""

    def __init__(self, n_wires: int, cnn_channels: list[int], fc_hidden: int) -> None:
        super().__init__()
        self.cnn = _CNNBackbone(cnn_channels[0], cnn_channels[1:3])
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        self.n_wires = n_wires
        self.quantum = tq.QuantumModule(
            n_wires=n_wires,
            device="cpu",
            record_mode=True,
        )
        self.quantum.register_module(
            "encoder",
            tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            ),
        )
        self.quantum.register_module(
            "random_layer",
            tq.RandomLayer(
                n_ops=30,
                wires=list(range(n_wires)),
            ),
        )
        self.quantum.register_module(
            "measure",
            tq.MeasureAll(tq.PauliZ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.cnn(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)  # shape (B,4)

        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        self.quantum.encoder(qdev, out)
        self.quantum.random_layer(qdev)
        out_q = self.quantum.measure(qdev)
        return out_q


class QLSTM(nn.Module):
    """LSTM cell where each gate is realised by a small quantum circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__(n_wires=n_wires, device="cpu")
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

        @tq.static_support
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=x.shape[0],
                device=x.device,
            )
            self.encoder(qdev, x)
            for wire in range(self.n_wires):
                self.params[wire](qdev, wires=wire)
                if wire < self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                else:
                    tqf.cnot(qdev, wires=[wire, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self._QLayer(n_qubits)
        self.input = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output = self._QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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


class _QuantumSeqEncoder(nn.Module):
    """Sequence tagger that uses a quantum LSTM if qubits > 0."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_hidden: int,
        tagset_size: int,
        qlstm_qubits: int,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if qlstm_qubits > 0:
            self.lstm = QLSTM(embedding_dim, lstm_hidden, n_qubits=qlstm_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, lstm_hidden, batch_first=False)
        self.hidden2tag = nn.Linear(lstm_hidden, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (B, T, embed_dim)
        embeds = embeds.permute(1, 0, 2)  # (T, B, embed_dim)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=2)


class QuantumHybridNAT(nn.Module):
    """Quantum‑enhanced hybrid model for image‑to‑sequence tasks.

    Parameters
    ----------
    n_wires : int
        Number of quantum wires for the image encoder.
    qlstm_qubits : int
        Number of qubits for the quantum LSTM (zero for classical).
    cnn_channels : list[int]
        Channel configuration for the CNN backbone.
    fc_hidden : int
        Size of the hidden layer in the fully‑connected head.
    lstm_hidden : int
        Hidden size of the LSTM (or quantum LSTM).
    vocab_size : int
        Vocabulary size for the embedding layer.
    embedding_dim : int
        Dimension of the word embeddings.
    tagset_size : int
        Number of output tags.
    """

    def __init__(
        self,
        n_wires: int,
        qlstm_qubits: int,
        cnn_channels: list[int],
        fc_hidden: int,
        lstm_hidden: int,
        vocab_size: int,
        embedding_dim: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.image_encoder = _QuantumImageEncoder(n_wires, cnn_channels, fc_hidden)
        self.seq_encoder = _QuantumSeqEncoder(
            vocab_size, embedding_dim, lstm_hidden, tagset_size, qlstm_qubits
        )

    def forward(
        self,
        image: torch.Tensor,
        sentence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return quantum‑processed image features and sequence tag logits."""
        img_out = self.image_encoder(image)
        seq_out = self.seq_encoder(sentence)
        return img_out, seq_out
