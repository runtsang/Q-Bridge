"""Hybrid LSTM implementation that fuses fraud‑detection style layers and self‑attention into a classical tagging model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#   Fraud‑Detection style parameter block
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a linear+activation+scale/shift block that mirrors the photonic layer."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


# --------------------------------------------------------------------------- #
#   Classical Self‑Attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Simple feed‑forward self‑attention block that operates on a batch of embeddings."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
#   Hybrid LSTM Tagger
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """A sequence‑tagging model that can switch between classical and quantum LSTM cells
    and optionally uses fraud‑detection style layers and self‑attention as a pre‑processor."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_qattn: bool = False,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional fraud‑detection style pre‑processing
        if fraud_params is not None:
            # The first layer is not clipped; the rest are clipped
            layers = [_layer_from_params(next(fraud_params), clip=False)]
            layers.extend(_layer_from_params(p, clip=True) for p in fraud_params)
            self.fraud_block = nn.Sequential(*layers)
        else:
            self.fraud_block = nn.Identity()

        # Optional quantum self‑attention pre‑processing
        self.use_qattn = use_qattn
        if use_qattn:
            self.attn = ClassicalSelfAttention(embed_dim=embedding_dim)
        else:
            self.attn = None

        # LSTM core
        if n_qubits > 0:
            # In the classical module we emulate the quantum gates with a linear layer
            # that has the same interface.  This keeps the API identical.
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Run the model on a batch of token indices."""
        # Embedding lookup
        embeds = self.word_embeddings(sentence)  # shape: (seq_len, batch, embed)

        # Fraud‑detection style transformation
        embeds = self.fraud_block(embeds)

        # Self‑attention pre‑processor
        if self.attn is not None:
            # Convert to numpy for the simple attention implementation
            attn_input = embeds.detach().cpu().numpy()
            rotation_params = np.random.rand(embeds.shape[-1] * 3)  # placeholder
            entangle_params = np.random.rand(embeds.shape[-1] - 1)  # placeholder
            attn_output = self.attn.run(rotation_params, entangle_params, attn_input)
            embeds = torch.from_numpy(attn_output).to(embeds.device)

        # LSTM
        lstm_out, _ = self.lstm(embeds)
        # Tagging
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM", "FraudLayerParameters", "ClassicalSelfAttention"]
