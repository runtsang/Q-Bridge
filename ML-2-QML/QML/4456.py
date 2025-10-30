import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QuantumAttention(nn.Module):
    """Multi‑head attention where each head is realised by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine = nn.Linear(embed_dim, embed_dim)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3).contiguous()
        heads = []
        for i in range(self.num_heads):
            head = x[:, i, :, :]  # (batch, seq, d_k)
            flat = head.reshape(-1, self.d_k)
            qdev = self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=flat.size(0), device=flat.device)
            out = self.q_layer(flat, qdev)
            out = out.reshape(batch, seq, self.d_k)
            heads.append(out)
        out = torch.stack(heads, dim=1)  # (batch, heads, seq, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outs.append(out)
        out = torch.stack(outs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and a quantum feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 n_qlayers: int,
                 dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridQuantumHead(nn.Module):
    """Hybrid head that maps a scalar to a quantum expectation."""
    def __init__(self, in_features: int, shots: int = 1024):
        super().__init__()
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()
    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        return np.array([expectation(result)])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            thetas = x.mean(dim=1).cpu().numpy()
        exp = self.run(thetas)
        return torch.tensor(exp, dtype=x.dtype, device=x.device)

class QuantumCNNFeatureExtractor(nn.Module):
    """CNN followed by a quantum feed‑forward."""
    def __init__(self, in_channels: int, embed_dim: int, n_qubits: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16 * 7 * 7, embed_dim)
        self.q_ffn = FeedForwardQuantum(embed_dim, embed_dim, n_qubits)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.q_ffn(x)

class LSTMTaggerQuantum(nn.Module):
    """Sequence tagging with a quantum LSTM."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)
    def __init__(self, embed_dim: int, hidden_dim: int, tagset_size: int, n_qubits: int):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(embed_dim, n_qubits)
        self.linear_input = nn.Linear(embed_dim, n_qubits)
        self.linear_update = nn.Linear(embed_dim, n_qubits)
        self.linear_output = nn.Linear(embed_dim, n_qubits)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        hx = lstm_out[:, -1, :]  # last hidden state
        f = torch.sigmoid(self.forget(self.linear_forget(hx)))
        i = torch.sigmoid(self.input(self.linear_input(hx)))
        g = torch.tanh(self.update(self.linear_update(hx)))
        o = torch.sigmoid(self.output(self.linear_output(hx)))
        cx = f * g
        hx = o * torch.tanh(cx)
        logits = self.hidden2tag(hx)
        return F.log_softmax(logits, dim=-1)

class QTransformerTorchGen136(nn.Module):
    """Unified transformer with optional quantum sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_cnn: bool = False,
                 use_lstm: bool = False,
                 lstm_hidden_dim: int = 128,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 **kwargs):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        if use_cnn:
            self.cnn = QuantumCNNFeatureExtractor(in_channels=3, embed_dim=embed_dim, n_qubits=n_qubits_ffn)
        else:
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        if n_qubits_transformer > 0:
            self.transformer = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim,
                                          num_heads,
                                          ffn_dim,
                                          n_qubits_transformer,
                                          n_qubits_ffn,
                                          n_qlayers,
                                          dropout=dropout) for _ in range(num_blocks)]
            )
        else:
            self.transformer = nn.Sequential(
                *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
            )
        if use_lstm:
            if n_qubits_transformer > 0:
                self.lstm_tagger = LSTMTaggerQuantum(embed_dim, lstm_hidden_dim, num_classes, n_qubits=n_qubits_transformer)
            else:
                self.lstm_tagger = LSTMTagger(embed_dim, lstm_hidden_dim, num_classes)
        else:
            self.classifier = HybridQuantumHead(num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            x = self.cnn(x).unsqueeze(1)
        else:
            x = self.token_embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        if self.use_lstm:
            return self.lstm_tagger(x)
        else:
            x = x.mean(dim=1)
            return self.classifier(x)

__all__ = ["QTransformerTorchGen136"]
