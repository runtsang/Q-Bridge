import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

# ----------------------------------------------------------------------
# Data generation – quantum version (identical to the original QML seed)
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset for quantum regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Quantum self‑attention (simple RX + CNOT circuit)
# ----------------------------------------------------------------------
class QuantumSelfAttention(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    def __init__(self, n_wires: int):
        super().__init__()
        self.q_layer = self.QLayer(n_wires)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        return self.q_layer(x, qdev)

# ----------------------------------------------------------------------
# Quantum multi‑head attention
# ----------------------------------------------------------------------
class MultiHeadAttentionQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, n_wires: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.q_layer = self.QLayer(n_wires)
        self.dropout = nn.Dropout(dropout)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        batch, seq, _ = x.shape
        heads = []
        for h in range(self.num_heads):
            proj = x[..., h * self.d_k : (h + 1) * self.d_k].reshape(batch * seq, self.d_k)
            qdev_bsz = proj.shape[0]
            qdev_local = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                          bsz=qdev_bsz,
                                          device=proj.device)
            out = self.q_layer(proj, qdev_local)
            heads.append(out.reshape(batch, seq, self.d_k))
        out = torch.cat(heads, dim=-1)
        return self.combine(out)

# ----------------------------------------------------------------------
# Quantum feed‑forward block
# ----------------------------------------------------------------------
class FeedForwardQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_wires)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        batch, seq, _ = x.shape
        proj = x.reshape(batch * seq, -1)
        qdev_bsz = proj.shape[0]
        qdev_local = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                      bsz=qdev_bsz,
                                      device=proj.device)
        out = self.q_layer(proj, qdev_local)
        out = out.reshape(batch, seq, -1)
        out = self.linear1(out)
        out = self.dropout(out)
        return self.linear2(F.relu(out))

# ----------------------------------------------------------------------
# Quantum transformer block
# ----------------------------------------------------------------------
class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_q_wires: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, n_q_wires, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_q_wires, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        attn_out = self.attn(x, qdev)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x, qdev)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------------------------
# Hybrid regression model – quantum transformer backbone
# ----------------------------------------------------------------------
class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum transformer‑based regression model.
    Mirrors the classical HybridRegressionModel but replaces the
    transformer blocks with quantum‑enabled ones.
    """
    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        n_q_wires: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_dim)
        # Simple sinusoidal positional encoding (classical) for simplicity
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                      n_q_wires, dropout)
             for _ in range(num_blocks)]
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, seq_len, num_features)

        Returns
        -------
        torch.Tensor
            Shape (batch,)
        """
        bsz = state_batch.shape[0]
        # Create a shared quantum device for all blocks
        qdev = tq.QuantumDevice(n_wires=self.blocks[0].attn.q_layer.n_wires,
                                bsz=bsz,
                                device=state_batch.device)
        x = self.input_proj(state_batch)
        # Add deterministic positional bias
        pos = torch.arange(state_batch.shape[1], dtype=torch.float32,
                           device=state_batch.device).unsqueeze(0).unsqueeze(-1)
        x = x + pos * 1e-3
        for block in self.blocks:
            x = block(x, qdev)
        return self.head(x.mean(dim=1)).squeeze(-1)
