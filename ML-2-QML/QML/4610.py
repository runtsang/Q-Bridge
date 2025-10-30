import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import math
import numpy as np

class QuantumSelfAttention(nn.Module):
    """Quantum‑enhanced multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, n_wires: int = 8, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_wires = n_wires
        self.dropout = nn.Dropout(dropout)
        self.k_proj = nn.Linear(embed_dim, n_wires)
        self.q_proj = nn.Linear(embed_dim, n_wires)
        self.v_proj = nn.Linear(embed_dim, n_wires)
        self.combine = nn.Linear(n_wires, embed_dim)
        self.qlayer = self.QLayer(n_wires)

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch*seq, device=x.device)
        out = self.qlayer(x.reshape(batch*seq, dim), qdev)
        return out.reshape(batch, seq, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        k_q = self._apply_quantum(k)
        q_q = self._apply_quantum(q)
        v_q = self._apply_quantum(v)
        scores = torch.matmul(q_q, k_q.transpose(-2, -1)) / math.sqrt(self.n_wires)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_q)
        return self.combine(out)

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realized by a quantum module."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_wires = n_wires
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.qlayer = self.QLayer(n_wires)

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.size()
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch*seq, device=x.device)
        out = self.qlayer(x.reshape(batch*seq, dim), qdev)
        out = out.reshape(batch, seq, self.n_wires)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """Hybrid transformer block with quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumSelfAttention(embed_dim, num_heads, n_wires, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                             (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TextClassifierQuantum(nn.Module):
    """Transformer‑based text classifier with quantum submodules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 n_wires: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_wires, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.blocks(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

class QLSTMQuantum(nn.Module):
    """LSTM cell where each gate is a quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_wires):
                if w == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[w, 0])
                else:
                    tqf.cnot(qdev, wires=[w, w+1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.forget = self.QLayer(n_wires)
        self.input_gate = self.QLayer(n_wires)
        self.update = self.QLayer(n_wires)
        self.output_gate = self.QLayer(n_wires)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_wires)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))

class LSTMTaggerQuantum(nn.Module):
    """Sequence tagger that can use a quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_wires: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_wires)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

def SelfAttention(**kwargs):
    """Factory returning a quantum self‑attention block."""
    return QuantumSelfAttention(**kwargs)

__all__ = [
    "QuantumSelfAttention",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
    "QLSTMQuantum",
    "LSTMTaggerQuantum",
    "SelfAttention",
]
