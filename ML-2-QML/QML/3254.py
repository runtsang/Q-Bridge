from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Base attention
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, mask: Optional[torch.Tensor] = None, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attention_weights: Optional[torch.Tensor] = None
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
    def compute_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, value)
        return out, scores
    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, batch_size: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.split_heads(query)
        k = self.split_heads(key)
        v = self.split_heads(value)
        out, self.attention_weights = self.compute_attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, mask: Optional[torch.Tensor] = None, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.final_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        if _!= self.embed_dim:
            raise ValueError(f"Input embedding ({_}) does not match layer embedding size ({self.embed_dim})")
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.final_proj(out)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where projections are passed through a quantum circuit."""
    class QuantumLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, mask: Optional[torch.Tensor] = None, use_bias: bool = False, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.quantum_layer = self.QuantumLayer()
        self.q_device = q_device
        self.final_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        if _!= self.embed_dim:
            raise ValueError(f"Input embedding ({_}) does not match layer embedding size ({self.embed_dim})")
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.final_proj(out)
    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device or tq.QuantumDevice(self.quantum_layer.n_wires, bsz=head.size(0), device=head.device)
                head_outputs.append(self.quantum_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

# Feed‑forward base
class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realized by a quantum module."""
    class QuantumLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.quantum_layer = self.QuantumLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.fc1 = nn.Linear(n_qubits, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.quantum_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.fc1(self.dropout(out))
        return self.fc2(F.relu(out))

# Transformer block
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block with classical sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attention = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.feedforward = FeedForwardClassical(embed_dim, ffn_dim, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.feedforward(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum‑enabled transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_transformer: int, n_qubits_ffn: int, n_qlayers: int, q_device: Optional[tq.QuantumDevice] = None, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attention = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        if n_qubits_ffn > 0:
            self.feedforward = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.feedforward = FeedForwardClassical(embed_dim, ffn_dim, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.feedforward(x)
        return self.norm2(x + self.dropout(ffn_out))

# Positional encoding
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# Transformer‑based classifier (quantum enabled)
class TextClassifier(nn.Module):
    """Transformer‑based text classifier with optional quantum sub‑modules."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int, ffn_dim: int, num_classes: int, dropout: float = 0.1,
                 n_qubits_transformer: int = 0, n_qubits_ffn: int = 0, n_qlayers: int = 1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            q_device = q_device or tq.QuantumDevice(n_wires=max(n_qubits_transformer, n_qubits_ffn))
            blocks = [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device=q_device, dropout=dropout) for _ in range(num_blocks)]
        else:
            blocks = [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        self.encoder = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_enc(tokens)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.output_layer(x)

# Quantum LSTM
class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QuantumLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_gate = self.QuantumLayer(n_qubits)
        self.input_gate = self.QuantumLayer(n_qubits)
        self.update_gate = self.QuantumLayer(n_qubits)
        self.output_gate = self.QuantumLayer(n_qubits)
        self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_proj(combined)))
            i = torch.sigmoid(self.input_gate(self.input_proj(combined)))
            g = torch.tanh(self.update_gate(self.update_proj(combined)))
            o = torch.sigmoid(self.output_gate(self.output_proj(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)
    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

# LSTM‑based classifier (quantum)
class LSTMClassifierQuantum(nn.Module):
    """LSTM‑based text classifier that optionally uses quantum gates."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1, n_qubits: int = 0) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        if n_qubits > 0:
            self.lstm_module = QLSTM(embed_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm_module = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(x)
        if isinstance(self.lstm_module, QLSTM):
            outputs, _ = self.lstm_module(embeds)
        else:
            outputs, _ = self.lstm_module(embeds)
        last_hidden = outputs[:, -1, :]
        out = self.dropout(last_hidden)
        return self.output_layer(out)

# Hybrid classifier (quantum)
class HybridTextClassifier(nn.Module):
    """Hybrid transformer / LSTM text classifier with optional quantum sub‑modules."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int, ffn_dim: int, num_classes: int,
                 backbone: str = 'transformer', lstm_hidden_dim: Optional[int] = None,
                 n_qubits_transformer: int = 0, n_qubits_ffn: int = 0, n_qlayers: int = 1,
                 q_device: Optional[tq.QuantumDevice] = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone_type = backbone
        if backbone == 'transformer':
            self.backbone = TextClassifier(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim, num_classes, dropout,
                                          n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device)
        elif backbone == 'lstm':
            if lstm_hidden_dim is None:
                raise ValueError('lstm_hidden_dim must be specified for lstm backbone')
            self.backbone = LSTMClassifierQuantum(vocab_size, embed_dim, lstm_hidden_dim, num_classes, dropout, n_qubits=n_qubits_transformer)
        else:
            raise ValueError('backbone must be either "transformer" or "lstm"')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
    "QLSTM",
    "LSTMClassifierQuantum",
    "HybridTextClassifier",
]
