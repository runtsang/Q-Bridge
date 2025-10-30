import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# ---------- Quantum Fully‑Connected Head ----------
class QFCModel(tq.QuantumModule):
    """Quantum fully‑connected model inspired by the Quantum‑NAT paper."""
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

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# ---------- Transformer Encoder (classical) ----------
class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

# ---------- Hybrid CNN + Transformer + Quantum Head ----------
class HybridCNNTransformerQuantumClassifier(nn.Module):
    """
    Combines a convolutional backbone, a transformer encoder, and a quantum
    fully‑connected head (QFCModel). The model is suitable for binary
    image classification tasks.
    """
    def __init__(
        self,
        img_channels: int = 3,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 256,
        patch_size: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.5),
        )

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # After CNN, feature map channels = 15
        patch_dim = 15 * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim)

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
        )

        # Linear projection to quantum input size (4 features)
        self.proj = nn.Linear(embed_dim, 4)

        # Quantum head
        self.quantum_head = QFCModel()

        # Final classifier
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.cnn(x)  # shape: [B, 15, H', W']
        # Patch extraction
        x = self.unfold(x)  # shape: [B, patch_dim, L]
        x = x.permute(0, 2, 1)  # [B, L, patch_dim]
        # Embed patches
        x = self.patch_embed(x)  # [B, L, embed_dim]
        # Positional encoding
        x = self.pos_encoder(x)
        # Transformer encoder
        x = self.transformer(x)  # [B, L, embed_dim]
        # Global average pooling over sequence
        x = x.mean(dim=1)  # [B, embed_dim]
        # Projection to quantum input
        x = self.proj(x).unsqueeze(-1).unsqueeze(-1)  # [B, 4, 1, 1] to match QFCModel input shape
        # Quantum fully‑connected head
        x = self.quantum_head(x)  # [B, 4]
        # Final classification logits
        logits = self.classifier(x)  # [B, num_classes]
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = [
    "QFCModel",
    "PositionalEncoding",
    "TransformerBlockClassical",
    "TransformerEncoder",
    "HybridCNNTransformerQuantumClassifier",
]
