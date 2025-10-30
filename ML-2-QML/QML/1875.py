import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

from. import QTransformerTorch__gen060 as ml

class MultiHeadAttentionQuantum(ml.MultiHeadAttentionBase):
    """Multi-head attention where each token is processed by a variational quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for i in range(self.n_qubits - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            tqf.cnot(q_device, wires=[self.n_qubits - 1, 0])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_qubits: int = 8,
                 q_device: tq.QuantumDevice = None,
                 **kwargs) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)
        self.kproj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.qproj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vproj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _quantum_transform(self, x: torch.Tensor) -> torch.Tensor:
        B, L, E = x.shape
        tokens = x.view(-1, E)
        inputs = tokens[:, :self.n_qubits]
        qdev = self.q_device.copy(bsz=inputs.size(0), device=inputs.device)
        out = self.q_layer(inputs, qdev)
        if self.n_qubits < E:
            pad = torch.zeros((out.size(0), E - self.n_qubits), device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out.view(B, L, E)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x_q = self._quantum_transform(x)
        k = self.kproj(x_q)
        q = self.qproj(x_q)
        v = self.vproj(x_q)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        attn_out = self.attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.out_proj(attn_out)

class FeedForwardQuantum(ml.FeedForwardBase):
    """Feed-forward subblock that uses a variational quantum circuit followed by linear layers."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
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

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _quantum_transform(self, x: torch.Tensor) -> torch.Tensor:
        B, L, E = x.shape
        tokens = x.view(-1, E)
        inputs = tokens[:, :self.n_qubits]
        qdev = self.q_device.copy(bsz=inputs.size(0), device=inputs.device)
        out = self.q_layer(inputs, qdev)
        if self.n_qubits < E:
            pad = torch.zeros((out.size(0), E - self.n_qubits), device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out.view(B, L, E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._quantum_transform(x)
        out = self.linear1(self.dropout(F.relu(x_q)))
        return self.linear2(out)

class TransformerBlockQuantum(ml.TransformerBlockBase):
    """Transformer block that combines quantum attention and quantum feed-forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_attn: int = 8,
                 n_qubits_ffn: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_qubits=n_qubits_attn)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                      n_qubits=n_qubits_ffn, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridTextClassifier(ml.TextClassifier):
    """Text classifier that can switch between classical and quantum transformer blocks."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_qubits_attn: int = 8,
                 n_qubits_ffn: int = 8,
                 **kwargs) -> None:
        super().__init__(vocab_size, embed_dim, num_heads, num_blocks,
                         ffn_dim, num_classes, dropout)
        if use_quantum:
            self.layers = nn.ModuleList(
                [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                         n_qubits_attn=n_qubits_attn,
                                         n_qubits_ffn=n_qubits_ffn,
                                         dropout=dropout)
                 for _ in range(num_blocks)]
            )
