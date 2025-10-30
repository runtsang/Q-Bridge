import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""
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

class MultiHeadAttentionQuantum(nn.Module):
    """Quantum multi‑head attention that processes each head through a small variational circuit."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a variational quantum circuit followed by classical linear layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self._build_q_layer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def _build_q_layer(self, n_qubits: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
                )
                self.parameters = nn.ParameterList(
                    [nn.Parameter(torch.randn(1)) for _ in range(n_qubits)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for i, gate in enumerate(self.parameters):
                    tqf.rx(q_device, wires=[i], params=gate)
                return self.measure(q_device)
        return QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        out = []
        for i in range(seq_len):
            token = x[:, i, :]  # (batch, embed_dim)
            qdev = self.q_device.copy(bsz=batch, device=token.device)
            for w in range(self.n_qubits):
                tqf.rx(qdev, wires=[w], params=token[:, w % self.n_qubits])
            token_out = self.q_layer(token, qdev)
            out.append(token_out)
        out = torch.stack(out, dim=1)  # (batch, seq_len, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """Quantum transformer block combining quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QCNNQuantumModule(tq.QuantumModule):
    """Quantum circuit mimicking the QCNN architecture with convolution and pooling layers."""
    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = nn.ParameterList(
            [
                nn.Parameter(torch.randn(n_qubits // 2 * 3))
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x: (batch, n_qubits)
        for i in range(self.n_qubits):
            tqf.rx(q_device, wires=[i], params=x[:, i])
        for layer_params in self.params:
            for i in range(0, self.n_qubits, 2):
                idx = i // 2
                theta0 = layer_params[idx * 3]
                theta1 = layer_params[idx * 3 + 1]
                theta2 = layer_params[idx * 3 + 2]
                tqf.rx(q_device, wires=[i], params=theta0)
                tqf.ry(q_device, wires=[i + 1], params=theta1)
                tqf.cx(q_device, wires=[i, i + 1])
                tqf.ry(q_device, wires=[i + 1], params=theta2)
                tqf.cx(q_device, wires=[i + 1, i])
        return tq.MeasureAll(tq.PauliZ)(q_device)

class HybridQuantumTransformer(nn.Module):
    """Hybrid quantum‑classical transformer that uses a QCNN‑style quantum module as a feature extractor."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 8,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 16,
        num_classes: int = 2,
        dropout: float = 0.1,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoder(embed_dim)
        self.qcnn_module = QCNNQuantumModule(n_qubits, n_layers=3)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        tokens = self.token_embedding(x)
        x = self.pos_encoding(tokens)
        x_flat = x.view(batch * seq_len, x.size(2))
        q_device = tq.QuantumDevice(n_wires=x.size(2), bsz=batch * seq_len, device=x.device)
        feats = self.qcnn_module(x_flat, q_device)  # (batch*seq_len, embed_dim)
        feats = feats.view(batch, seq_len, x.size(2))
        x = self.transformer(feats)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "PositionalEncoder",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "QCNNQuantumModule",
    "HybridQuantumTransformer",
]
