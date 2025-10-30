"""Hybrid transformer classifier with optional autoencoder and quantum classifier head.

This module mirrors the classical QTransformerTorch implementation but
adds support for a pre‑processing autoencoder and a quantum classifier
head constructed with Qiskit.  The design is intentionally lightweight
so that the class can be used as a drop‑in replacement for the
original ``TextClassifier`` while still exposing the additional
quantum interfaces.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports – the seed modules are assumed to be on the PYTHONPATH
from QTransformerTorch import PositionalEncoder, TransformerBlockClassical
from Autoencoder import Autoencoder, AutoencoderConfig
from QuantumClassifierModel import build_classifier_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class HybridTransformerClassifier(nn.Module):
    """
    A transformer‑based text classifier that can optionally prepend an
    autoencoder pre‑processor and append a Qiskit quantum classifier head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        *,
        use_autoencoder: bool = False,
        autoencoder_config: Optional[AutoencoderConfig] = None,
        use_quantum_classifier: bool = False,
        qiskit_circuit_depth: int = 3,
        qiskit_num_qubits: int = 4,
    ) -> None:
        super().__init__()

        # Optional autoencoder pre‑processor
        if use_autoencoder:
            if autoencoder_config is None:
                raise ValueError("autoencoder_config must be provided when use_autoencoder is True")
            self.autoencoder = Autoencoder(
                input_dim=autoencoder_config.input_dim,
                latent_dim=autoencoder_config.latent_dim,
                hidden_dims=autoencoder_config.hidden_dims,
                dropout=autoencoder_config.dropout,
            )
        else:
            self.autoencoder = None

        # Classic transformer backbone
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        # Optional quantum classifier head
        self.quantum_head: Optional[SamplerQNN] = None
        if use_quantum_classifier:
            circuit, enc_params, var_params, observables = build_classifier_circuit(
                qiskit_num_qubits, qiskit_circuit_depth
            )
            sampler = StatevectorSampler()
            self.quantum_head = SamplerQNN(
                circuit=circuit,
                weight_params=var_params,
                input_params=enc_params,
                interpret=lambda x: x,
                output_shape=len(observables),
                sampler=sampler,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len) containing token indices for
            text classification.  If an autoencoder is enabled, ``x`` should
            be a continuous feature set of shape (batch, feature_dim).

        Returns
        -------
        torch.Tensor
            Logits or probabilities depending on ``num_classes``.
        """
        # Pre‑process with autoencoder if present
        if self.autoencoder is not None:
            # Autoencoder expects continuous features
            x = self.autoencoder.encode(x.float())

        # Token embedding and positional encoding
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)

        # Transformer stack
        x = self.transformers(x)

        # Pooling, dropout, and classification head
        x = self.dropout(x.mean(dim=1))
        logits = self.classifier(x)

        # Quantum classifier head (if configured)
        if self.quantum_head is not None:
            # The SamplerQNN expects a numpy array of shape (batch, num_inputs)
            # We use the logits as the input vector to the quantum circuit
            q_input = logits.detach().cpu().numpy()
            q_out = self.quantum_head(q_input)
            # Convert back to torch tensor
            logits = torch.tensor(q_out, device=x.device, dtype=x.dtype)

        return logits


__all__ = [
    "HybridTransformerClassifier",
]
