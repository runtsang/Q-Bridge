"""Quantum regression model that fuses a parameterised circuit with a classical transformer."""
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Optional

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states and labels."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega0 = np.zeros(2 ** num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2 ** num_wires, dtype=complex)
        omega1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns quantum state tensors and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumEncoder(tq.QuantumModule):
    """Encodes classical features into a quantum state using RX gates."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(num_wires)
            ]
        )
        # Optional random layer for added expressivity
        self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(num_wires)))

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        # x shape [B, N]
        self.encoder(qdev, x)
        self.random_layer(qdev)

class HybridQuantumLayer(tq.QuantumModule):
    """Stack of parameterised rotations followed by a simple entangling block."""
    def __init__(self, num_wires: int, n_layers: int = 3, entangle_depth: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.entangle_depth = entangle_depth
        self.rotation_params = nn.ParameterList(
            [nn.Parameter(torch.randn(num_wires, 3)) for _ in range(n_layers)]
        )
        # Entanglement pattern: a chain of CNOTs
        self.entanglers = nn.ModuleList(
            [nn.ModuleList([tq.CNOT() for _ in range(entangle_depth)]) for _ in range(n_layers)]
        )

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        # x shape [B, N]
        for layer in range(self.n_layers):
            for wire in range(self.num_wires):
                params = self.rotation_params[layer][wire]
                tq.RX(params[0], wires=[wire])(qdev)
                tq.RY(params[1], wires=[wire])(qdev)
                tq.RZ(params[2], wires=[wire])(qdev)
            # Entanglement
            for _ in range(self.entangle_depth):
                for wire in range(self.num_wires - 1):
                    tq.CNOT(wires=[wire, wire + 1])(qdev)
                tq.CNOT(wires=[self.num_wires - 1, 0])(qdev)

class TransformerBlockClassical(nn.Module):
    """Standard transformer block used after the quantum measurement."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumRegressionEnhanced(tq.QuantumModule):
    """Hybrid regression model that uses a quantum encoder, a parameterised layer, and an optional classical transformer."""
    def __init__(
        self,
        num_wires: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.encoder = QuantumEncoder(num_wires)
        self.hybrid = HybridQuantumLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        if use_transformer:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(num_wires, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.hybrid(qdev, state_batch)
        features = self.measure(qdev)
        if not isinstance(self.transformers, nn.Identity):
            features = self.transformers(features)
        return self.head(features).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumEncoder",
    "HybridQuantumLayer",
    "TransformerBlockClassical",
    "QuantumRegressionEnhanced",
]
