"""Hybrid quantum sampler autoencoder combining variational circuit, sampler, and transformer interpretation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# ----------------------------------------------------------------------
# Classical transformer block (used in the interpret function)
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------------------------
# Interpret function that processes measurement results via a transformer
# ----------------------------------------------------------------------
class TransformerInterpret(nn.Module):
    def __init__(self, embed_dim: int = 2, num_heads: int = 4, ffn_dim: int = 128):
        super().__init__()
        self.transformer = TransformerBlock(embed_dim, num_heads, ffn_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, shots, qubits)
        batch, shots, qubits = x.shape
        x = x.reshape(batch * shots, qubits)
        x = x.unsqueeze(1)  # sequence length 1
        x = self.transformer(x)
        x = x.squeeze(1)
        logits = self.classifier(x)
        return F.softmax(logits, dim=-1)

# ----------------------------------------------------------------------
# Hybrid quantum sampler
# ----------------------------------------------------------------------
def HybridSamplerAutoEncoder() -> SamplerQNN:
    # Parameter vectors for inputs and trainable weights
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    # Variational circuit using a simple Ry/CX pattern
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    # Optional: richer ansatz
    # qc = RealAmplitudes(2, reps=3).bind_parameters({... })

    sampler = StatevectorSampler()
    interpret = TransformerInterpret(embed_dim=2, num_heads=4, ffn_dim=128)

    return SamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
        interpret=interpret
    )

__all__ = ["HybridSamplerAutoEncoder"]
