"""
Classical implementation of a hybrid sequence tagger that optionally
applies a fraud‑detection style linear network to each word embedding
before feeding the sequence to an LSTM.  The class name `QLSTM` is kept
identical to the original anchor so that existing import paths remain
stable, while the added `fraud_params` argument allows the model to
switch between a pure classical LSTM and a quantum‑enhanced one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------
# Fraud‑detection style layer building utilities (classical analogue)
# --------------------------------------------------------------------
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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool, input_dim: int) -> nn.Module:
    """
    Build a 2‑by‑2 linear layer that can be reshaped to the required
    input_dim.  For simplicity we only support input_dim == 2; otherwise
    a ValueError is raised.
    """
    if input_dim!= 2:
        raise ValueError("FraudProcessor currently supports only 2‑dimensional inputs.")
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    input_dim: int = 2,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False, input_dim=input_dim)]
    modules.extend(_layer_from_params(layer, clip=True, input_dim=input_dim) for layer in layers)
    modules.append(nn.Linear(2, 2))  # keep dimensionality consistent
    return nn.Sequential(*modules)

# --------------------------------------------------------------------
# Hybrid LSTM implementation
# --------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Classical LSTM cell that can optionally prepend a fraud‑detection style
    preprocessing network.  The same class name is reused in the quantum
    module; only the preprocessing implementation differs.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        fraud_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_fraud = fraud_params is not None

        if self.use_fraud:
            layers = list(fraud_params)
            self.fraud_processor = build_fraud_detection_program(layers[0], layers[1:])
        else:
            self.fraud_processor = None

        # Standard LSTM gates
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        # Optional fraud preprocessing
        if self.fraud_processor is not None:
            processed = []
            for x in inputs.unbind(dim=0):
                processed.append(self.fraud_processor(x.unsqueeze(0)))
            inputs = torch.cat(processed, dim=0)

        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can be configured to use the hybrid
    QLSTM cell with optional fraud preprocessing.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        fraud_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            fraud_params=fraud_params,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger", "FraudLayerParameters", "build_fraud_detection_program"]
