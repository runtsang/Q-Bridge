import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum‑enhanced patch encoder using a random 2‑qubit circuit."""
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

class QCNNQuantumBlock(tq.QuantumModule):
    """QCNN‑style block using a random quantum circuit per patch."""
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        x = x.view(bsz, 28, 28)
        patches = []
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
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
                self.random_layer(qdev, data)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        features = torch.cat(patches, dim=1)
        return self.linear(features)

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Quantum‑enhanced multi‑head attention (placeholder using a random layer)."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=max(num_heads, embed_dim))
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.q_device.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = self.q_device.copy(bsz=bsz, device=x.device)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        return measurement.view(bsz, 1, self.embed_dim)

class FeedForwardQuantum(tq.QuantumModule):
    """Quantum feed‑forward network using a random layer and measurement."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=x.device)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        out = self.linear1(measurement)
        out = self.linear2(self.dropout(F.relu(out)))
        return out

class TransformerBlockQuantum(tq.QuantumModule):
    """Transformer block with quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridQuanvolutionModel(tq.QuantumModule):
    """Hybrid model combining quantum quanvolution, QCNN, and transformer."""
    def __init__(self,
                 num_classes: int = 10,
                 transformer_layers: int = 2,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 ffn_dim: int = 128,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.qcnn = QCNNQuantumBlock(input_dim=4 * 14 * 14, embed_dim=embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim,
                                      num_heads,
                                      ffn_dim,
                                      n_qubits_transformer,
                                      n_qubits_ffn)
              for _ in range(transformer_layers)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        features = self.qcnn(features)
        features = features.unsqueeze(1)  # [B,1,embed_dim]
        x = self.transformers(features)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionModel"]
