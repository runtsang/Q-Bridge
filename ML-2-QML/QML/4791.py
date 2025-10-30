"""Quantum‑augmented quanvolution + transformer model.

The quantum implementation replaces the classical layers with TorchQuantum
modules, enabling a direct comparison of parameter counts, training dynamics
and expressibility.  The public API mirrors the classical version so that
experiment scripts can swap between the two with minimal changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.quantum as tq  # type: ignore
import torch.quantum.functional as tqf  # type: ignore


# --------------------------------------------------------------------------- #
#  Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Apply a 2‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        device = x.device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=device)

        # reshape to 28×28 patches
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
                self.random_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
#  Quantum Transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi‑head attention whose projections are processed by a quantum layer."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int = 8) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.parameters:
                gate(qdev)
            # entangle all qubits with a simple ring of CNOTs
            for i in range(self.n_qubits - 1):
                tqf.cnot(qdev, [i, i + 1])
            tqf.cnot(qdev, [self.n_qubits - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qlayer = self.QLayer(n_qubits=8)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        # linear projections
        proj = nn.Linear(self.embed_dim, self.embed_dim).to(x.device)
        q = proj(x)
        k = proj(x)
        v = proj(x)

        # quantum processing per head
        out = []
        for token in q.unbind(dim=1):
            qdev = tq.QuantumDevice(self.qlayer.n_qubits, bsz=token.size(0), device=token.device)
            head_out = self.qlayer(token, qdev)
            out.append(head_out)
        out = torch.stack(out, dim=1)

        # simple soft‑max attention with dropout
        scores = torch.matmul(out, out.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, out)


class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward block realised by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.parameters:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.qlayer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(self.qlayer.n_qubits, bsz=token.size(0), device=token.device)
            out.append(self.qlayer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum Transformer block combining Q‑Attention and Q‑FFN."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Quantum hybrid model
# --------------------------------------------------------------------------- #
class HybridQuanvolutionTransformerEstimator(tq.QuantumModule):
    """Quantum‑augmented hybrid model.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    out_channels : int
        Number of channels after the quanvolution filter.
    embed_dim : int
        Embedding dimension for transformer layers.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension in the feed‑forward sub‑network.
    regression_hidden : int
        Hidden size of the regression head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        regression_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilterQuantum(in_channels, out_channels)
        self.proj = nn.Linear(out_channels * 14 * 14, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, regression_hidden),
            nn.ReLU(),
            nn.Linear(regression_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        seq = self.proj(feats).unsqueeze(1)
        seq = self.transformer(seq)
        return self.regressor(seq.mean(dim=1))


# --------------------------------------------------------------------------- #
#  Utility: Qiskit estimator
# --------------------------------------------------------------------------- #
def EstimatorQNN():
    """Return a Qiskit EstimatorQNN instance for regression."""
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.primitives import StatevectorEstimator

    param_input = Parameter("x")
    param_weight = Parameter("w")

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(param_input, 0)
    qc.rx(param_weight, 0)

    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[param_input],
        weight_params=[param_weight],
        estimator=estimator,
    )


__all__ = [
    "QuanvolutionFilterQuantum",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "HybridQuanvolutionTransformerEstimator",
    "EstimatorQNN",
]
