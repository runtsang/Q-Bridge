"""
Hybrid regression model for quantum training.

The module exposes:
  * `RegressionDataset` – quantum version of the dataset.
  * `HybridQModel` – a quantum encoder + measurement + classical transformer head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation (quantum version)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate superposition states and scalar labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of states to generate.

    Returns
    -------
    states : np.ndarray
        Shape (samples, 2**num_wires), dtype=complex.
    labels : np.ndarray
        Shape (samples,), dtype=float32.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * np.eye(1, 2 ** num_wires, 0)[0] + \
                    np.exp(1j * phis[i]) * np.sin(thetas[i]) * np.eye(1, 2 ** num_wires, -1)[0]
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Quantum dataset yielding state vectors and scalar targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Positional encoding (copied from QTransformerTorch)
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Transformer block (quantum‑aware)
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that uses a quantum layer for the attention projections.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(num_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(num_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.parameters, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
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
        self.q_layer = self.QLayer(n_qubits)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical attention path
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Quantum feed‑forward
        batch, seq_len, _ = x.shape
        # Treat each token as a separate quantum state
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=batch * seq_len, device=x.device)
        # Flatten tokens to feed into quantum layer
        flat = x.reshape(batch * seq_len, -1)
        quantum_out = self.q_layer(flat, qdev)
        quantum_out = quantum_out.reshape(batch, seq_len, -1)
        ffn_out = self.ffn(quantum_out)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Hybrid quantum model
# --------------------------------------------------------------------------- #
class HybridQModel(tq.QuantumModule):
    """
    Quantum encoder + measurement + classical transformer head.
    """
    def __init__(
        self,
        num_wires: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        n_qubits_transformer: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_wires = num_wires
        # Quantum encoder that maps a state vector to a register of size `num_wires`
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational layer
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        # Measurement to obtain classical features
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical transformer head
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits=n_qubits_transformer)
                for _ in range(num_blocks)
            ]
        )
        # Final regression head
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (B, 2**num_wires), dtype=complex.

        Returns
        -------
        torch.Tensor
            Predicted scalar, shape (B,).
        """
        bsz = state_batch.shape[0]
        # Quantum device for the encoder
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # (B, num_wires)
        # Embed to transformer dimension
        embed = nn.Linear(self.n_wires, embed_dim).to(state_batch.device)
        seq = embed(features).unsqueeze(1)  # treat each sample as a single token
        seq = self.pos_encoder(seq)
        seq = self.transformer(seq)
        pooled = seq.mean(dim=1)
        return self.fc(pooled).squeeze(-1)

__all__ = ["RegressionDataset", "HybridQModel"]
