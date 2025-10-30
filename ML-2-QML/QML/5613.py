import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import torchquantum as tq

class ParamScaledLinear(nn.Module):
    """Linear layer with learnable scaling and shift, inspired by fraud‑detection layers."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Parameter(torch.ones(out_features))
        self.shift = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale + self.shift

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
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

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑augmented multi‑head attention."""
    class QHead(tq.QuantumModule):
        def __init__(self, d_k: int):
            super().__init__()
            self.n_wires = d_k
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
            )
            self.rxs = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(d_k)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.rxs:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.d_k = embed_dim // num_heads
        self.q_heads = nn.ModuleList([self.QHead(self.d_k) for _ in range(num_heads)])
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.combine = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, embed_dim = x.size()
        x_split = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        head_outputs = []
        for i, head in enumerate(self.q_heads):
            head_out = []
            for token in x_split[:, i, :, :].unbind(dim=1):
                qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
                out = head(token, qdev)
                head_out.append(out)
            head_out = torch.stack(head_out, dim=1)
            head_outputs.append(head_out)
        out = torch.stack(head_outputs, dim=1)  # (batch, heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.combine(out)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "ry", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.rys = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.rys:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_attention: int = 8, n_qubits_ffn: int = 8,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class ConvFilter(nn.Module):
    """Classical convolution that mimics the quantum filter shape."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies a random two‑qubit circuit to 2×2 patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
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
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement)
        return torch.cat(patches, dim=1)

class HybridTransformerClassifier(nn.Module):
    """
    Hybrid transformer that supports text or image inputs.
    Image mode can use a classical convolution or the quantum QuanvolutionFilter.
    Transformer blocks can be configured to use quantum attention and feed‑forward.
    """
    def __init__(
        self,
        mode: str = "text",
        vocab_size: int = 30522,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        use_quantum_filter: bool = False,
        n_qubits_attention: int = 8,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "text":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        elif mode == "image":
            if use_quantum_filter:
                self.embedding = QuanvolutionFilter()
            else:
                self.embedding = ConvFilter()
        else:
            raise ValueError("mode must be 'text' or 'image'")

        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum_attention or use_quantum_ffn:
                block = TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    n_qubits_attention=n_qubits_attention,
                    n_qubits_ffn=n_qubits_ffn,
                    dropout=dropout,
                )
            else:
                block = TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    n_qubits_attention=n_qubits_attention,
                    n_qubits_ffn=n_qubits_ffn,
                    dropout=dropout,
                )
            self.transformer_blocks.append(block)

        self.dropout = nn.Dropout(dropout)
        self.classifier = ParamScaledLinear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "text":
            x = self.embedding(x)  # (batch, seq_len, embed_dim)
        else:
            x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)
        for block in self.transformer_blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

__all__ = [
    "ParamScaledLinear",
    "PositionalEncoder",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "ConvFilter",
    "QuanvolutionFilter",
    "HybridTransformerClassifier",
]
