from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

from QTransformerTorch import (
    TransformerBlockQuantum,
    TransformerBlockClassical,
    PositionalEncoder,
)
from Autoencoder import Autoencoder, AutoencoderConfig
from SelfAttention import SelfAttention

class QuantumClassifierHead(tq.QuantumModule):
    """
    Simple quantum classifier head that maps a classical feature vector
    to a set of qubit rotations and returns the expectation values of
    Pauli‑Z as logits.
    """

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)


class HybridTransformerClassifier(nn.Module):
    """
    Quantum‑enhanced transformer classifier that unifies the classical
    transformer blocks, optional quantum attention, a quantum feed‑forward
    module, an optional auto‑encoder, and a quantum classifier head.
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
        n_qubits_transformer: int = 0,
        n_qubits_classifier: int = 0,
        n_qlayers: int = 1,
        use_autoencoder: bool = False,
        autoencoder_config: AutoencoderConfig | None = None,
        use_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Transformer blocks – quantum if requested
        if n_qubits_transformer > 0:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_transformer,
                        n_qlayers,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )

        self.dropout = nn.Dropout(dropout)

        # Optional auto‑encoder
        if use_autoencoder:
            cfg = autoencoder_config or AutoencoderConfig(
                input_dim=embed_dim,
                latent_dim=max(4, embed_dim // 4),
                hidden_dims=(embed_dim, max(4, embed_dim // 2)),
                dropout=dropout,
            )
            self.autoencoder = Autoencoder(cfg)
        else:
            self.autoencoder = None

        # Optional quantum self‑attention (not used in forward)
        self.self_attention = SelfAttention() if use_self_attention else None

        # Classifier head
        if n_qubits_classifier > 0:
            self.linear_to_qubits = nn.Linear(embed_dim, n_qubits_classifier)
            self.classifier = QuantumClassifierHead(n_qubits_classifier)
            self.q_device = tq.QuantumDevice(n_wires=n_qubits_classifier)
        else:
            self.classifier = nn.Linear(
                embed_dim, num_classes if num_classes > 2 else 1
            )
            self.q_device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)  # (B, L, E)
        tokens = self.pos_embedding(tokens)

        if self.autoencoder is not None:
            batch, seq_len, embed_dim = tokens.shape
            flat = tokens.view(-1, embed_dim)
            encoded = self.autoencoder.encode(flat)
            decoded = self.autoencoder.decode(encoded)
            tokens = decoded.view(batch, seq_len, embed_dim)

        x = self.transformers(tokens)

        # Global pooling and classification
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)

        if isinstance(self.classifier, QuantumClassifierHead):
            qubits = self.linear_to_qubits(pooled)
            logits = self.classifier(qubits, self.q_device)
            return logits
        else:
            return self.classifier(pooled)
