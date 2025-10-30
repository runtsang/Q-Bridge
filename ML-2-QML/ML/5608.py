import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KernalAnsatz(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Kernel that can be classical or quantum."""
    def __init__(self, gamma: float = 1.0, use_quantum: bool = False) -> None:
        super().__init__()
        if use_quantum:
            raise NotImplementedError("Quantum kernel not available in classical module.")
        self.ansatz = KernalAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a, b, gamma: float = 1.0, use_quantum: bool = False):
    kernel = Kernel(gamma, use_quantum)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QFCModel(nn.Module):
    """CNN followed by a fully connected projection."""
    def __init__(self, use_quantum_fc: bool = False, n_qubits: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        self.use_quantum_fc = use_quantum_fc
        if use_quantum_fc:
            raise NotImplementedError("Quantum FC layer not available in classical module.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = features.view(x.shape[0], -1)
        out = self.fc(flattened)
        return self.norm(out)

class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, v), scores
    def downstream(self, q, k, v, batch_size, mask=None):
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi-head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)
    def forward(self, x: torch.Tensor, mask=None):
        batch_size, _, _ = x.size()
        q, k, v = self.q_linear(x), self.k_linear(x), self.v_linear(x)
        return self.combine(self.downstream(q, k, v, batch_size, mask))

class FeedForwardBase(nn.Module):
    """Shared interface for feed-forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardClassical(FeedForwardBase):
    """Two-layer perceptron feed-forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed-forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__(embed_dim, num_heads)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim)
    def forward(self, x: torch.Tensor):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    """Transformer-based text classifier supporting quantum submodules."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1,
                 n_qubits_transformer: int = 0, n_qubits_ffn: int = 0, n_qlayers: int = 1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            raise NotImplementedError("Quantum transformer not available in classical module.")
        blocks = [TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor):
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

class QuantumKernelEnhanced:
    """Unified interface for kernel evaluation, feature extraction, and classification."""
    def __init__(self, kernel_gamma: float = 1.0, use_quantum_kernel: bool = False,
                 use_quantum_fc: bool = False, use_quantum_transformer: bool = False,
                 n_qubits_fc: int = 4, n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8, n_qlayers: int = 1,
                 vocab_size: int = 10000, embed_dim: int = 128, num_heads: int = 8,
                 num_blocks: int = 4, ffn_dim: int = 256, num_classes: int = 10,
                 dropout: float = 0.1):
        self.kernel = Kernel(gamma=kernel_gamma, use_quantum=use_quantum_kernel)
        self.qfc = QFCModel(use_quantum_fc=use_quantum_fc, n_qubits=n_qubits_fc)
        self.transformer = TextClassifier(vocab_size=vocab_size, embed_dim=embed_dim,
                                          num_heads=num_heads, num_blocks=num_blocks,
                                          ffn_dim=ffn_dim, num_classes=num_classes,
                                          dropout=dropout,
                                          n_qubits_transformer=0, n_qubits_ffn=0, n_qlayers=1)
        self.use_quantum_kernel = use_quantum_kernel
        self.use_quantum_fc = use_quantum_fc
        self.use_quantum_transformer = use_quantum_transformer
    def compute_kernel(self, a, b):
        return kernel_matrix(a, b, gamma=1.0, use_quantum=self.use_quantum_kernel)
    def extract_features(self, x):
        return self.qfc(x)
    def classify(self, x):
        return self.transformer(x)
    def __repr__(self):
        return f"<QuantumKernelEnhanced kernel={self.kernel} qfc={self.qfc} transformer={self.transformer}>"

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QFCModel",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TextClassifier",
    "QuantumKernelEnhanced"
]
