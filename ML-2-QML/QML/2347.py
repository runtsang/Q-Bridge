import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchquantum as tq
import torchquantum.functional as tqf

class QCNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.head(x)

class QuantumProjection(tq.QuantumModule):
    def __init__(self, n_qubits: int, output_dim: int | None = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.output_dim = output_dim or n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.cnot_chain = [(i, i + 1) for i in range(n_qubits - 1)] + [(n_qubits - 1, 0)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=x.device)
        self.encoder(qdev, x)
        for w, gate in enumerate(self.params):
            gate(qdev, wires=w)
        for a, b in self.cnot_chain:
            tqf.cnot(qdev, wires=[a, b])
        out = self.measure(qdev)
        if self.output_dim < self.n_qubits:
            return out[:, :self.output_dim]
        return out

class QuantumAttention(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.pre_proj = nn.Linear(embed_dim, 8)
        self.q_proj = QuantumProjection(8, self.d_k)
        self.k_proj = QuantumProjection(8, self.d_k)
        self.v_proj = QuantumProjection(8, self.d_k)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        x_flat = x.reshape(batch * seq, dim)
        q_in = self.pre_proj(x_flat)
        k_in = self.pre_proj(x_flat)
        v_in = self.pre_proj(x_flat)
        Q = self.q_proj(q_in).reshape(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(k_in).reshape(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(v_in).reshape(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn = torch.matmul(scores, V)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(attn)

class FeedForwardQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pre_proj = nn.Linear(embed_dim, n_qubits)
        self.q_proj = QuantumProjection(n_qubits, n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        x_flat = x.reshape(batch * seq, dim)
        q_in = self.pre_proj(x_flat)
        q_out = self.q_proj(q_in)
        out = self.linear1(q_out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = out.reshape(batch, seq, dim)
        return out

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridQCNNTransformerQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int,
                 num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.qcnn = QCNNFeatureExtractor(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x_flat = x.reshape(batch * seq, 8)
        qcnn_out = self.qcnn(x_flat)
        x = qcnn_out.reshape(batch, seq, -1)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

def HybridQCNNTransformerQuantumFactory(embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int,
                                        num_classes: int, dropout: float = 0.1) -> HybridQCNNTransformerQuantum:
    return HybridQCNNTransformerQuantum(embed_dim, num_heads, ffn_dim, num_blocks, num_classes, dropout)

__all__ = ["QCNNFeatureExtractor", "QuantumProjection", "QuantumAttention", "FeedForwardQuantum",
           "TransformerBlockQuantum", "PositionalEncoder", "HybridQCNNTransformerQuantum",
           "HybridQCNNTransformerQuantumFactory"]
