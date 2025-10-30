"""Quantum‑enhanced hybrid LSTM tagger.

This module defines QLSTMHybrid that uses a quantum convolutional
encoder followed by a quantum LSTM cell. The quantum layers are
implemented with torchquantum and only activate when ``n_qubits > 0``.
When ``n_qubits == 0`` the class falls back to a classical implementation
for compatibility with environments lacking a quantum backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLSTMHybrid(nn.Module):
    """Quantum hybrid LSTM tagger.

    Parameters
    ----------
    embedding_dim : int
        Size of the word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Vocabulary size for the embedding layer.
    tagset_size : int
        Number of target tags.
    n_qubits : int
        Number of qubits used in each quantum sub‑module.
    """

    class QConvEncoder(tq.QuantumModule):
        """Quantum convolutional encoder that projects embeddings to qubit states."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Input of shape (batch, seq_len, n_wires).

            Returns
            -------
            torch.Tensor
                Encoded qubit probabilities of shape (batch, seq_len, n_wires).
            """
            bsz, seq_len, n_wires = x.shape
            qdev = tq.QuantumDevice(n_wires=n_wires, bsz=bsz * seq_len, device=x.device)
            flat = x.reshape(bsz * seq_len, n_wires)
            self.encoder(qdev, flat)
            for idx, gate in enumerate(self.params):
                gate(qdev, wires=idx)
            out = self.measure(qdev)
            return out.reshape(bsz, seq_len, n_wires)

    class QLayer(tq.QuantumModule):
        """Quantum gate used inside the quantum LSTM cell."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Input of shape (batch, n_wires).

            Returns
            -------
            torch.Tensor
                Output probabilities of shape (batch, n_wires).
            """
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
            self.encoder(qdev, x)
            for idx, gate in enumerate(self.params):
                gate(qdev, wires=idx)
            return self.measure(qdev)

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
        self.n_qubits = n_qubits

        # Choose number of qubits for encoder and LSTM gates
        self.n_wires = n_qubits if n_qubits > 0 else hidden_dim

        if n_qubits > 0:
            # Quantum encoder
            self.conv_encoder = self.QConvEncoder(self.n_wires)
            # Quantum LSTM gates
            self.forget_gate = self.QLayer(self.n_wires)
            self.input_gate = self.QLayer(self.n_wires)
            self.update_gate = self.QLayer(self.n_wires)
            self.output_gate = self.QLayer(self.n_wires)

            # Linear projections from classical concatenated state to qubit space
            self.linear_forget = nn.Linear(embedding_dim + hidden_dim, self.n_wires)
            self.linear_input = nn.Linear(embedding_dim + hidden_dim, self.n_wires)
            self.linear_update = nn.Linear(embedding_dim + hidden_dim, self.n_wires)
            self.linear_output = nn.Linear(embedding_dim + hidden_dim, self.n_wires)

            # Linear projection for embedding to qubit space
            self.embed_to_qubits = nn.Linear(embedding_dim, self.n_wires)
        else:
            # Fallback to classical LSTM
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=False,
            )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape (seq_len, batch) containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag scores of shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)

        if self.n_qubits > 0:
            seq_len, batch, _ = embeds.shape
            # Project embeddings to qubit space for the encoder
            embed_proj = self.embed_to_qubits(embeds)  # (seq_len, batch, n_wires)
            embed_proj = embed_proj.permute(1, 0, 2)  # (batch, seq_len, n_wires)
            encoded = self.conv_encoder(embed_proj)  # (batch, seq_len, n_wires)

            # Initialize hidden/cell states
            hx = torch.zeros(batch, self.hidden_dim, device=embeds.device)
            cx = torch.zeros(batch, self.hidden_dim, device=embeds.device)

            outputs = []
            for t in range(seq_len):
                x_t = encoded[t]  # (batch, n_wires)
                combined = torch.cat([x_t, hx], dim=1)  # (batch, n_wires + hidden_dim)
                f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
                i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
                g = torch.tanh(self.update_gate(self.linear_update(combined)))
                o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
                cx = f * cx + i * g
                hx = o * torch.tanh(cx)
                outputs.append(hx.unsqueeze(0))
            lstm_out = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
        else:
            lstm_out, _ = self.lstm(embeds)

        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, "
            f"n_qubits={self.n_qubits})"
        )


__all__ = ["QLSTMHybrid"]
