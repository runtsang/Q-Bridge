"""Hybrid self‑attention module that fuses classical attention with an LSTM‑style gate.

The module implements a drop‑in replacement for the seed `SelfAttention.py`.  It
provides a single `SelfAttentionHybrid` class that can operate purely
classically or, if `n_qubits>0`, generate rotation and entanglement
parameters via a lightweight neural network that mimics the quantum
parameter‑generation logic of the QML seed.

Key features:
* Classical query‑key‑value attention with softmax normalisation.
* Optional LSTM‑style gating (forget/input/update/output) that modulates
  the attention scores.
* A small MLP that produces rotation/entanglement parameters when a quantum
  backend is requested, enabling seamless switching between pure‑classical
  and hybrid modes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHybrid(nn.Module):
    """
    Classical self‑attention block with optional LSTM‑style gating and
    quantum‑parameter generation.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_qubits : int, default 0
        If >0, a lightweight MLP will generate rotation and entanglement
        parameters that can be fed into a quantum backend.  The class
        itself remains fully classical.
    """

    def __init__(self, embed_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits

        # LSTM‑style gating layers
        self.forget_gate = nn.Linear(2 * embed_dim, embed_dim)
        self.input_gate = nn.Linear(2 * embed_dim, embed_dim)
        self.update_gate = nn.Linear(2 * embed_dim, embed_dim)
        self.output_gate = nn.Linear(2 * embed_dim, embed_dim)

        # Parameter generator for quantum backend (if requested)
        if n_qubits > 0:
            # Two linear layers produce rotation and entanglement angles
            self.rotation_gen = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 3),
                nn.ReLU(),
                nn.Linear(embed_dim * 3, n_qubits),
            )
            self.entangle_gen = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 3),
                nn.ReLU(),
                nn.Linear(embed_dim * 3, n_qubits - 1),
            )
        else:
            self.rotation_gen = None
            self.entangle_gen = None

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the hybrid attention block.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape
            (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape

        # Compute query, key, value
        query = inputs
        key = inputs
        value = inputs

        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.embed_dim)
        scores = F.softmax(scores, dim=-1)

        # Optional LSTM‑style gating
        # Concatenate query and key for gate computation
        combined = torch.cat([query, key], dim=-1)  # (batch, seq_len, 2*embed_dim)
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        g = torch.tanh(self.update_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))

        # Modulate scores with gate output
        gate = o * torch.tanh(f * g)
        scores = scores * gate

        # Weighted sum
        out = torch.bmm(scores, value)
        return out

    def generate_quantum_params(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Produce rotation and entanglement parameters for a quantum backend.
        Only available when `n_qubits > 0`.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] | None
            Rotation parameters of shape (batch, seq_len, n_qubits) and
            entanglement parameters of shape (batch, seq_len, n_qubits-1),
            or None if `n_qubits == 0`.
        """
        if self.n_qubits == 0:
            return None

        batch, seq_len, _ = inputs.shape
        flat = inputs.reshape(-1, self.embed_dim)

        rot = self.rotation_gen(flat)  # (batch*seq_len, n_qubits)
        ent = self.entangle_gen(flat)  # (batch*seq_len, n_qubits-1)

        rot = rot.reshape(batch, seq_len, self.n_qubits)
        ent = ent.reshape(batch, seq_len, self.n_qubits - 1)
        return rot, ent

__all__ = ["SelfAttentionHybrid"]
