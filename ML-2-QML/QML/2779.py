import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import torchquantum as tq
import torchquantum.functional as tqf

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

class MultiHeadAttentionQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, d_k: int):
            super().__init__()
            self.d_k = d_k
            self.n_wires = d_k
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(d_k)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.parameters:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, device: Optional[torch.device] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.device = device or torch.device('cpu')
        self.q_layer = self.QLayer(self.d_k)
        self.combine_heads = nn.Linear(num_heads * self.d_k, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        heads = []
        for head_idx in range(self.num_heads):
            token_vecs = x[:, head_idx, :, :]
            flat = token_vecs.reshape(-1, self.d_k)
            qdev = tq.QuantumDevice(n_wires=self.d_k, bsz=flat.size(0), device=self.device)
            out = self.q_layer(flat, qdev)
            out = out.reshape(batch_size, seq_len, self.d_k)
            heads.append(out)
        out = torch.stack(heads, dim=2)
        out = out.view(batch_size, seq_len, self.num_heads * self.d_k)
        return self.combine_heads(out)

class FeedForwardQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.parameters:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1, device: Optional[torch.device] = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.device = device or torch.device('cpu')
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        outputs = []
        for i in range(seq_len):
            token = x[:, i, :]
            proj = token[:, :self.n_qubits]
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=proj.size(0), device=self.device)
            out = self.q_layer(proj, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, device=self.device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=embed_dim, dropout=dropout, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumPoolingLayer(tq.QuantumModule):
    def __init__(self, n_qubits: int, device: torch.device):
        super().__init__()
        self.n_qubits = n_qubits
        self.device = device
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pool = F.avg_pool2d(x, 6).view(bsz, -1)
        pool = pool[:, :self.n_qubits]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=self.device)
        self.encoder(qdev, pool)
        for gate in self.parameters:
            gate(qdev)
        return self.measure(qdev)

class QuantumHybridNAT(tq.QuantumModule):
    """
    Quantum implementation of the hybrid Quantum‑NAT architecture.
    Replaces key linear projections with variational quantum circuits.
    """
    def __init__(
        self,
        mode: str = 'cnn',
        vocab_size: int = 30522,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        num_classes: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.device = device or torch.device('cpu')

        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

        # Transformer encoder
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout=0.1, device=self.device) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Quantum pooling layer for CNN mode
        self.q_pool = QuantumPoolingLayer(n_qubits=4, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'cnn':
            bsz = x.shape[0]
            feat = self.features(x)
            flat = feat.view(bsz, -1)
            out = self.fc(flat)
            q_out = self.q_pool(x)
            return self.norm(out + q_out)
        else:
            tokens = self.token_embedding(x)
            tokens = self.pos_encoder(tokens)
            for blk in self.transformer_blocks:
                tokens = blk(tokens)
            x = tokens.mean(dim=1)
            return self.dropout(self.classifier(x))

__all__ = ["QuantumHybridNAT"]
