import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Optional

# ------------------------------------------------------------------
# Quantum quanvolution filter – 2×2 patches mapped to a 4‑dim output
# ------------------------------------------------------------------
class QuanvolutionFilterQuantum(tq.QuantumModule):
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

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
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
                self.encoder(q_device, data)
                self.q_layer(q_device)
                measurement = self.measure(q_device)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


# ------------------------------------------------------------------
# Quantum multi‑head attention (simplified: single quantum layer per head)
# ------------------------------------------------------------------
class MultiHeadAttentionQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_wires=embed_dim)
        self.q_device = q_device
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        qdev = self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=batch_size, device=x.device)
        q_out = self.q_layer(x, qdev)
        return self.combine(q_out)


# ------------------------------------------------------------------
# Quantum feed‑forward block
# ------------------------------------------------------------------
class FeedForwardQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.parameters):
            gate(qdev, wires=wire)
        meas = self.measure(qdev)
        out = self.linear1(self.dropout(meas))
        return self.linear2(F.relu(out))


# ------------------------------------------------------------------
# Quantum transformer block
# ------------------------------------------------------------------
class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=embed_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ------------------------------------------------------------------
# Main hybrid model – quantum variant
# ------------------------------------------------------------------
class HybridNATModel(tq.QuantumModule):
    def __init__(
        self,
        use_quanvolution: bool = True,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if use_quanvolution:
            self.feature_extractor = QuanvolutionFilterQuantum()
            self.feature_dim = 4 * 14 * 14
        else:
            self.feature_extractor = None  # fallback to classical conv if desired
            self.feature_dim = 16 * 7 * 7

        self.fc = nn.Linear(self.feature_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None:
            features = self.feature_extractor(x)
        else:
            # placeholder: use raw input if no quantum feature extractor
            features = x
        x = self.fc(features)
        x = self.norm(x)
        x = x.unsqueeze(1)  # sequence length = 1
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


__all__ = ["HybridNATModel"]
