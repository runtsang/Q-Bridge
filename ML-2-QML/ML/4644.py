"""Hybrid classical LSTM with fraud‑style preprocessing and kernel feature mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters for a fraud‑style linear layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, clip: bool = True) -> nn.Module:
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()


def build_fraud_preprocessor(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Build a sequential fraud‑style preprocessing pipeline."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules += [_layer_from_params(layer, clip=True) for layer in layers]
    return nn.Sequential(*modules)


class KernelFeatureMapper(nn.Module):
    """Map a vector to a kernel space defined by reference vectors."""
    def __init__(self, references: Sequence[torch.Tensor], gamma: float = 1.0):
        super().__init__()
        self.register_buffer("refs", torch.stack(list(references)))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        diff = x.unsqueeze(1) - self.refs.unsqueeze(0)  # (batch, n_ref, dim)
        sq_norm = (diff * diff).sum(dim=-1)  # (batch, n_ref)
        return torch.exp(-self.gamma * sq_norm)  # (batch, n_ref)


class HybridQLSTM(nn.Module):
    """Hybrid LSTM that optionally uses quantum gates and includes fraud‑style
    preprocessing and kernel feature mapping."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        fraud_params: Tuple[FraudLayerParameters, Iterable[FraudLayerParameters]] | None = None,
        kernel_refs: Sequence[torch.Tensor] | None = None,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.preprocessor = (
            build_fraud_preprocessor(*fraud_params) if fraud_params else nn.Identity()
        )
        self.kernel_mapper = (
            KernelFeatureMapper(kernel_refs, kernel_gamma) if kernel_refs else nn.Identity()
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence: (seq_len, batch)
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        seq_len, batch, embed_dim = embeds.shape
        flat = embeds.view(seq_len * batch, embed_dim)
        flat = self.preprocessor(flat)
        flat = self.kernel_mapper(flat)
        flat = flat.view(seq_len, batch, -1)
        lstm_out, _ = self.lstm(flat)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQLSTM", "FraudLayerParameters", "build_fraud_preprocessor", "KernelFeatureMapper"]
