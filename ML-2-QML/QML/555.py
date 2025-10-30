import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridTransformer"]

class QuantumAttention(nn.Module):
    """Quantum‑enhanced multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_qubits: int = 8):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.n_qubits = n_qubits
        self.qc = tq.QuantumDevice(n_wires=n_qubits)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.cnot_pattern = [(i, i+1) for i in range(n_qubits-1)] + [(n_qubits-1, 0)]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        # Quantum post‑processing on each token
        out_q = torch.zeros_like(out)
        for i in range(batch):
            for j in range(seq_len):
                token = out[i, j, :].unsqueeze(0)
                qdev = self.qc.copy(bsz=1, device=token.device)
                self.encoder(qdev, token)
                for wire in range(self.n_qubits):
                    tqf.rx(qdev, wires=wire, params=token[0, wire % self.head_dim])
                for (c1, c2) in self.cnot_pattern:
                    tqf.cnot(qdev, wires=[c1, c2])
                out_q[i, j, :] = self.measure(qdev)
        return self.out_proj(out_q)

class QuantumFeedForward(nn.Module):
    """Quantum feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, n_qubits: int = 8):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_qubits = n_qubits
        self.qc = tq.QuantumDevice(n_wires=n_qubits)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = torch.zeros_like(x)
        for i in range(x.size(0)):
            token = x[i].unsqueeze(0)
            qdev = self.qc.copy(bsz=1, device=token.device)
            self.encoder(qdev, token)
            for wire in range(self.n_qubits):
                tqf.ry(qdev, wires=wire, params=token[0, wire % x.size(1)])
            out[i] = self.measure(qdev)
        return self.linear2(out)

class QuantumTransformerBlock(nn.Module):
    """Transformer block with quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, n_qubits: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout, n_qubits)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout, n_qubits)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridTransformer(nn.Module):
    """Hybrid transformer classifier with optional quantum sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 q_device: object | None = None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = QuantumPositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if n_qubits_transformer > 0:
                self.blocks.append(
                    QuantumTransformerBlock(embed_dim, num_heads, ffn_dim,
                                            dropout, n_qubits_transformer)
                )
            else:
                # classical fallback
                self.blocks.append(
                    nn.ModuleList([
                        nn.LayerNorm(embed_dim),
                        nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, ffn_dim),
                        nn.ReLU(),
                        nn.Linear(ffn_dim, embed_dim),
                    ])
                )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            if isinstance(block, QuantumTransformerBlock):
                x = block(x)
            else:
                attn_output, _ = block[1](x, x, x)
                x = block[0](x + self.dropout(attn_output))
                ffn_out = block[5](block[4](block[3](block[2](x))))
                x = block[2](x + self.dropout(ffn_out))
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def contrastive_loss(self,
                         logits: torch.Tensor,
                         labels: torch.Tensor,
                         lm_logits: torch.Tensor,
                         temperature: float = 0.07) -> torch.Tensor:
        """
        InfoNCE contrastive loss between transformer logits and a pre‑trained LM.
        """
        logits_norm = F.normalize(logits, dim=1)
        lm_norm = F.normalize(lm_logits, dim=1)
        logits_lm = torch.matmul(logits_norm, lm_norm.t()) / temperature
        loss = F.cross_entropy(logits_lm, labels)
        return loss
