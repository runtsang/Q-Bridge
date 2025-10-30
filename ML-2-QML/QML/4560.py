"""Hybrid quantum‑enhanced LSTM‑Transformer architecture.

The quantum implementation mirrors the classical design but replaces the
gates of the LSTM cell with small variational circuits and the
feed‑forward layers of the transformer with quantum modules.  All
components are defined in a single module for clarity and can be
instantiated through the ``HybridQLSTMTransformer`` class below.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
# Quantum LSTM cell – 2nd seed
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where each gate is a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections into the quantum space
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
            f = torch.sigmoid(
                self.forget(self.linear_forget(combined), tq.QuantumDevice(n_wires=self.n_qubits, bsz=combined.size(0), device=combined.device))
            )
            i = torch.sigmoid(
                self.input(self.linear_input(combined), tq.QuantumDevice(n_wires=self.n_qubits, bsz=combined.size(0), device=combined.device))
            )
            g = torch.tanh(
                self.update(self.linear_update(combined), tq.QuantumDevice(n_wires=self.n_qubits, bsz=combined.size(0), device=combined.device))
            )
            o = torch.sigmoid(
                self.output(self.linear_output(combined), tq.QuantumDevice(n_wires=self.n_qubits, bsz=combined.size(0), device=combined.device))
            )
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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


# --------------------------------------------------------------------------- #
# LSTMTagger – 2nd seed, with quantum fallback
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagger that uses quantum LSTM when requested."""
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
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(1)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
# Quantum feed‑forward – 3rd seed
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(nn.Module):
    """Two‑layer perceptron realised by a quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
# Quantum multi‑head attention – 3rd seed
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Attention that maps projections through a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, n_wires: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_wires)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)
        # linear projections (classical) to feed to quantum module
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_output = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_output.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_output, dim=1))
        proj = torch.stack(projections, dim=1)  # (B, S, H, d_k)
        # reshape for attention
        proj = proj.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        # classical attention on quantum‑embedded heads
        scores = torch.matmul(proj, proj.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, proj)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.combine_heads(out)


# --------------------------------------------------------------------------- #
# Quantum transformer block – 3rd seed
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(nn.Module):
    """Transformer block that may use quantum attention and/or FFN."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attention: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, n_qubits_attention, dropout=dropout
        )
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# Positional encoding – reused
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# TextClassifier – quantum‑aware
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """Transformer‑based text classifier that can switch to quantum blocks."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_attention: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_attention > 0 or n_qubits_ffn > 0:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_attention,
                        n_qubits_ffn,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockBase(
                        embed_dim, num_heads, ffn_dim, dropout
                    )
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Hybrid architecture – 1st + 2nd + 3rd seeds, quantum version
# --------------------------------------------------------------------------- #
class HybridQLSTMTransformer(nn.Module):
    """Combined quantum LSTM‑tagger, transformer‑classifier and regression head."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_classes: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        n_qubits_lstm: int = 0,
        n_qubits_attention: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        # Tagging head
        self.tagger = LSTMTagger(
            embedding_dim,
            hidden_dim,
            vocab_size,
            tagset_size,
            n_qubits=n_qubits_lstm,
        )
        # Classification head
        self.classifier = TextClassifier(
            vocab_size,
            embedding_dim,
            num_heads,
            num_blocks,
            ffn_dim,
            num_classes,
            dropout=0.1,
            n_qubits_attention=n_qubits_attention,
            n_qubits_ffn=n_qubits_ffn,
        )
        # Regression head (classical; can be swapped for a quantum regressor if desired)
        self.regressor = EstimatorNN(hidden_dim)

    def forward(
        self,
        sentence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        tags : torch.Tensor
            Log‑probabilities over tagset, shape (seq_len, tagset_size).
        logits : torch.Tensor
            Classification logits, shape (batch, num_classes).
        reg : torch.Tensor
            Regression prediction, shape (batch, 1).
        """
        tags = self.tagger(sentence)
        logits = self.classifier(sentence)
        with torch.no_grad():
            embeds = self.tagger.word_embeddings(sentence).unsqueeze(1)
            lstm_out, _ = self.tagger.lstm(embeds)
            hidden_mean = lstm_out.mean(dim=0)
        reg = self.regressor(hidden_mean)
        return tags, logits, reg


# --------------------------------------------------------------------------- #
# Base transformer block for classical fallback
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Classic transformer block used when no quantum gates are requested."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn2(self.dropout(F.relu(self.ffn1(x))))
        return self.norm2(x + self.dropout(ffn_out))


__all__ = [
    "QLSTM",
    "LSTMTagger",
    "EstimatorNN",
    "FeedForwardQuantum",
    "MultiHeadAttentionQuantum",
    "TransformerBlockQuantum",
    "TransformerBlockBase",
    "PositionalEncoder",
    "TextClassifier",
    "HybridQLSTMTransformer",
]
