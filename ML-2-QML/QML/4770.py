"""Hybrid regression model with quantum back‑bone.

The public class ``HybridRegression`` uses a quantum auto‑encoder
followed by a quantum transformer encoder.  The module mirrors the
classical API for consistency and preserves the data‑generation
logic.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    omega_0 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_0[0] = 1.0
    omega_1 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_1[-1] = 1.0
    thetas = 2 * torch.pi * torch.rand(samples)
    phis = 2 * torch.pi * torch.rand(samples)
    states = torch.zeros((samples, 2 ** num_wires), dtype=torch.cfloat)
    for i in range(samples):
        states[i] = torch.cos(thetas[i]) * omega_0 + torch.exp(1j * phis[i]) * torch.sin(thetas[i]) * omega_1
    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return self.states.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"states": self.states[idx], "target": self.labels[idx]}

class QuantumAutoencoder(tq.QuantumModule):
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.StatevectorEncoder(num_wires)
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.random_layer(qdev)
        return self.measure(qdev)

class MultiHeadAttentionQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_wires = embed_dim
        self.rx = tq.RX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class FeedForwardQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        for wire in range(self.n_qubits):
            self.ry(qdev, wires=wire)
        out = self.measure(qdev)
        out = self.linear1(self.dropout(out))
        return self.linear2(torch.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_transformer: int, n_qubits_ffn: int, n_qlayers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        attn_out = self.attn(qdev)
        x = self.norm1(attn_out)
        ffn_out = self.ffn(qdev)
        return self.norm2(x + self.dropout(ffn_out))

class HybridRegression(nn.Module):
    """Quantum hybrid regression model."""
    def __init__(
        self,
        num_wires: int,
        *,
        num_heads: int = 4,
        ffn_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_wires = num_wires
        self.autoencoder = QuantumAutoencoder(num_wires)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(
                embed_dim=num_wires,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                n_qubits_transformer=num_wires,
                n_qubits_ffn=num_wires,
                n_qlayers=1,
                dropout=dropout,
            ) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(num_wires, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        bsz = states.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=states.device)
        self.autoencoder.encoder(qdev, states)
        self.autoencoder.random_layer(qdev)
        features = None
        for block in self.transformers:
            features = block(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
