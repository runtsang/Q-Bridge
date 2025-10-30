import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumQuanvolutionFilter(tq.QuantumModule):
    'Quantum 2x2 quanvolution filter using a random 4‑qubit circuit.'
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {'input_idx': [0], 'func': 'ry', 'wires': [0]},
                {'input_idx': [1], 'func': 'ry', 'wires': [1]},
                {'input_idx': [2], 'func': 'ry', 'wires': [2]},
                {'input_idx': [3], 'func': 'ry', 'wires': [3]},
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

class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.parameters:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(self.n_qubits, bsz=token.shape[0], device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(out)
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_ffn: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumNATHybrid(tq.QuantumModule):
    'Quantum hybrid model: quantum quanvolution filter + quantum transformer encoder + linear head.'
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 256,
        num_classes: int = 10,
        n_qubits_ffn: int = 8,
    ):
        super().__init__()
        self.feature_extractor = QuantumQuanvolutionFilter()
        seq_len = 14 * 14
        self.proj = nn.Linear(4, embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_ffn) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.feature_extractor(x)
        seq = patches.view(x.size(0), 14 * 14, 4)
        seq = self.proj(seq)
        seq = self.transformers(seq)
        out = seq.mean(dim=1)
        return self.classifier(out)

__all__ = ['QuantumNATHybrid']
