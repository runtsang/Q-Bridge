import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict

class KernalAnsatz(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernalAnsatzQuantum(tq.QuantumModule):
    """Quantum kernel ansatz that encodes classical data."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class KernelQuantum(tq.QuantumModule):
    """Hybrid kernel that can be classical or quantum."""
    def __init__(self, gamma: float = 1.0, use_quantum: bool = True) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            self.q_device = tq.QuantumDevice(n_wires=4)
            self.ansatz = KernalAnsatzQuantum(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
        else:
            self.ansatz = KernalAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            self.ansatz(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])
        else:
            return self.ansatz(x, y).squeeze()

def kernel_matrix(a, b, gamma: float = 1.0, use_quantum: bool = True):
    kernel = KernelQuantum(gamma, use_quantum)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QFCModelQuantum(tq.QuantumModule):
    """Quantum fully connected model inspired by the Quantum-NAT paper."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
    def __init__(self, use_quantum_fc: bool = False, n_qubits: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)
        self.use_quantum_fc = use_quantum_fc
        if use_quantum_fc:
            self.q_encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.q_layer = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = features.view(x.shape[0], -1)
        out = self.fc(flattened)
        if self.use_quantum_fc:
            qdev = tq.QuantumDevice(n_wires=4, bsz=x.shape[0], device=x.device)
            self.q_encoder(qdev, out)
            self.q_layer(qdev, wires=0)
            out = self.measure(qdev)
        return self.norm(out)

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi-head attention that maps projections through quantum modules."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.gates = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, token, qdev):
            self.encoder(qdev, token)
            for wire, gate in enumerate(self.gates):
                gate(qdev, wires=wire)
                if wire < self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qlayer = self.QLayer(n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim)
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
    def forward(self, x: torch.Tensor, mask=None):
        batch_size, _, _ = x.size()
        q_out = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.qlayer.n_wires, bsz=token.size(0), device=token.device)
            q_out.append(self.qlayer(token, qdev))
        q_out = torch.stack(q_out, dim=1)
        return self.combine(self.downstream(q_out, q_out, q_out, batch_size, mask))

class FeedForwardClassical(FeedForwardBase):
    """Two-layer perceptron feed-forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(tq.QuantumModule):
    """Feed-forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.v = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, token, qdev):
            self.encoder(qdev, token)
            for wire, gate in enumerate(self.v):
                gate(qdev, wires=wire)
                if wire < self.n_qubits - 1:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor):
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_qubits, bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    """Transformer block that uses quantum attention and feed-forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int, n_qubits_ffn: int, n_qlayers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=n_qubits_transformer)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
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

class TextClassifierQuantum(tq.QuantumModule):
    """Transformer-based text classifier supporting quantum submodules."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1,
                 n_qubits_transformer: int = 0, n_qubits_ffn: int = 0, n_qlayers: int = 1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            blocks = [
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                       n_qubits_transformer, n_qubits_ffn,
                                       n_qlayers, dropout=dropout)
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim)
                for _ in range(num_blocks)
            ]
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
        self.kernel = KernelQuantum(gamma=kernel_gamma, use_quantum=use_quantum_kernel)
        self.qfc = QFCModelQuantum(use_quantum_fc=use_quantum_fc, n_qubits=n_qubits_fc)
        self.transformer = TextClassifierQuantum(vocab_size=vocab_size, embed_dim=embed_dim,
                                                 num_heads=num_heads, num_blocks=num_blocks,
                                                 ffn_dim=ffn_dim, num_classes=num_classes,
                                                 dropout=dropout,
                                                 n_qubits_transformer=n_qubits_transformer,
                                                 n_qubits_ffn=n_qubits_ffn, n_qlayers=n_qlayers)
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
    "KernalAnsatzQuantum",
    "KernelQuantum",
    "kernel_matrix",
    "QFCModelQuantum",
    "MultiHeadAttentionQuantum",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
    "QuantumKernelEnhanced"
]
