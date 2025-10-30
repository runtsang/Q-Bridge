"""Combined quantum sampler, transformer, and classifier pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

# Quantum imports
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

# Seed modules
from QTransformerTorch import TransformerBlockQuantum, PositionalEncoder
from QuantumClassifierModel import build_classifier_circuit


class SamplerQNNGen145(nn.Module):
    """
    Quantum‑enhanced pipeline that mirrors the classical counterpart.
    It constructs a parameterised quantum sampler, a transformer block
    with quantum attention, and a variational classifier circuit.
    """
    def __init__(
        self,
        seq_len: int = 10,
        embed_dim: int = 8,
        num_heads: int = 2,
        ffn_dim: int = 16,
        depth: int = 2,
        q_device=None,
    ):
        super().__init__()
        # Quantum sampler circuit
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        sampler = Sampler()
        self.sampler = QSamplerQNN(
            circuit=qc2,
            input_params=inputs2,
            weight_params=weights2,
            sampler=sampler,
        )

        # Transformer block with quantum attention
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer=embed_dim,
                    n_qubits_ffn=embed_dim,
                    n_qlayers=1,
                    q_device=q_device,
                    dropout=0.1,
                )
                for _ in range(depth)
            ]
        )

        # Linear projection from sampler output to transformer embedding
        self.embed_proj = nn.Linear(2, embed_dim)

        # Variational classifier circuit (classical wrapper around the quantum circuit)
        self.classifier, _, _, _ = build_classifier_circuit(num_qubits=embed_dim, depth=depth)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )

        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, 2) representing raw 2‑D features
               for each timestep.
        Returns:
            logits: Tensor of shape (batch, 2)
        """
        batch, seq_len, _ = x.shape

        # Sample probabilities for each timestep using the quantum sampler
        samp_logits = []
        for i in range(seq_len):
            samp_logits.append(self.sampler(x[:, i, :]))
        samp_seq = torch.stack(samp_logits, dim=1)  # (batch, seq_len, 2)

        # Project to transformer embedding dimension
        embed = self.embed_proj(samp_seq)

        # Positional encoding
        embed = self.pos_encoder(embed)

        # Transformer encoder
        trans_out = self.transformer(embed)  # (batch, seq_len, embed_dim)

        # Pooling and classification
        pooled = trans_out.mean(dim=1)  # (batch, embed_dim)
        logits = self.classifier(pooled)  # (batch, 2)

        return logits
