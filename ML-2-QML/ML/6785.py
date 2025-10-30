from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters that describe a photonic fraud‑layer, reused as a classical feature extractor."""
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
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2, bias=True)
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

        def forward(self, inputs: Tensor) -> Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_photonic_feature_extractor(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a classical sequential model that mirrors the photonic fraud‑detection circuit."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridQLSTM(nn.Module):
    """Classical LSTM cell with optional quantum‑style gate emulation."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        if n_qubits > 0:
            # emulate quantum gates with a small neural block
            self.forget_gate = nn.Sequential(nn.Linear(gate_dim, gate_dim), nn.Tanh())
            self.input_gate = nn.Sequential(nn.Linear(gate_dim, gate_dim), nn.Tanh())
            self.update_gate = nn.Sequential(nn.Linear(gate_dim, gate_dim), nn.Tanh())
            self.output_gate = nn.Sequential(nn.Linear(gate_dim, gate_dim), nn.Tanh())
        else:
            self.forget_gate = self.input_gate = self.update_gate = self.output_gate = None

    def forward(
        self,
        inputs: Tensor,
        states: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget_gate(f))
                i = torch.sigmoid(self.input_gate(i))
                g = torch.tanh(self.update_gate(g))
                o = torch.sigmoid(self.output_gate(o))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: Tensor,
        states: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridQLSTMTagger(nn.Module):
    """Sequence tagging model that prepends a photonic feature extractor."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.preprocessor = build_photonic_feature_extractor(input_params, layer_params)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: Tensor) -> Tensor:
        embeds = self.word_embeddings(sentence)
        preprocessed = self.preprocessor(embeds)
        lstm_out, _ = self.lstm(preprocessed.unsqueeze(1))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = [
    "FraudLayerParameters",
    "build_photonic_feature_extractor",
    "HybridQLSTM",
    "HybridQLSTMTagger",
]
