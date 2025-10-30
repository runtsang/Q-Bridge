"""Hybrid transformer with quantum‑enhanced attention, feed‑forward, LSTM,
and optional image processing via a quantum quanvolution filter.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# ----- Quantum modules -----

class QMultiHeadAttention(tq.QuantumModule):
    """Quantum‑enabled multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1)
        head_dim = self.d_k
        x_heads = x.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        outputs = []
        for h in range(self.num_heads):
            head = x_heads[:, h, :, :]
            qdev = tq.QuantumDevice(self.n_wires, bsz=batch_size, device=head.device)
            self.encoder(qdev, head)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            measurement = self.measure(qdev)
            outputs.append(measurement)
        out = torch.stack(outputs, dim=1)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.dropout(out)
        return out

class QFeedForward(tq.QuantumModule):
    """Quantum feed‑forward block."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1)
        outputs = []
        for b in range(batch_size):
            qdev = tq.QuantumDevice(self.n_qubits, bsz=1, device=x.device)
            token = x[b, :, :]
            self.encoder(qdev, token)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            measurement = self.measure(qdev)
            outputs.append(measurement)
        out = torch.stack(outputs, dim=0)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out

class QTransformerBlock(tq.QuantumModule):
    """Transformer block with quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QMultiHeadAttention(embed_dim, num_heads, dropout, n_qubits)
        self.ffn = QFeedForward(embed_dim, ffn_dim, n_qubits)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ----- Quantum LSTM -------------------------------------------

class QQuantumLSTM(tq.QuantumModule):
    """Quantum LSTM cell with gates realized by small quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_gate = self._create_gate()
        self.input_gate = self._create_gate()
        self.update_gate = self._create_gate()
        self.output_gate = self._create_gate()
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _create_gate(self):
        n_wires = self.n_qubits
        encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        measure = tq.MeasureAll(tq.PauliZ)
        gate = tq.QuantumModule()
        gate.n_wires = n_wires
        gate.encoder = encoder
        gate.params = params
        gate.measure = measure

        def forward(qdev, x):
            gate.encoder(qdev, x)
            for wire, g in enumerate(gate.params):
                g(qdev, wires=wire)
            return gate.measure(qdev)
        gate.forward = forward
        return gate

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# ----- Quantum Quanvolution -------------------------------------------

class QQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that processes 2×2 patches via a random quantum circuit."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# ----- Simple adjacency -------------------------------------------

def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-12)
    b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-12)
    return torch.matmul(a_norm, b_norm.transpose(-2, -1))

def _graph_convolution(states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    return torch.matmul(adjacency, states)

# ----- Hybrid Transformer -------------------------------------------

class HybridTransformer(tq.QuantumModule):
    """Quantum‑enhanced transformer that can process text or images,
    optionally run a quantum LSTM, and perform a graph‑convolution
    on the quantum‑encoded hidden states.
    """
    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_quanvolution: bool = False,
        use_lstm: bool = False,
        use_graph: bool = False,
        image_input: bool = False,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.use_lstm = use_lstm
        self.use_graph = use_graph
        self.image_input = image_input
        self.n_qubits = n_qubits

        if image_input:
            self.feature_extractor = QQuanvolutionFilter()
            feature_dim = 4 * 14 * 14
            self.token_embedding = nn.Linear(feature_dim, embed_dim)
        else:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for token inputs")
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoder = QPositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [QTransformerBlock(embed_dim, num_heads, ffn_dim, n_qubits, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)

        if use_lstm:
            self.lstm = QQuantumLSTM(embed_dim, embed_dim, n_qubits)
        else:
            self.lstm = None

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.image_input:
            features = self.feature_extractor(x)
            tokens = self.token_embedding(features.unsqueeze(1))
            seq_len = 1
        else:
            tokens = self.token_embedding(x)
            seq_len = x.size(1)

        x = self.pos_encoder(tokens)
        for block in self.transformer_blocks:
            x = block(x, mask)

        if self.lstm is not None:
            lstm_out, _ = self.lstm(x)
            x = lstm_out

        if self.use_graph:
            B, T, E = x.shape
            states = x.reshape(B * T, E)
            adj = _cosine_similarity(states, states)
            adj.fill_diagonal_(0)
            adj_sum = adj.sum(dim=-1, keepdim=True) + 1e-12
            adj = adj / adj_sum
            states = _graph_convolution(states, adj)
            x = states.reshape(B, T, E)

        x = self.dropout(x.mean(dim=1))
        logits = self.classifier(x)
        return logits
