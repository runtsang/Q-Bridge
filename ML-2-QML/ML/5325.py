"""Hybrid LSTM model integrating classical components.

This module defines a `HybridQLSTM` class that can optionally apply
fraud‑detection preprocessing, a convolutional filter, and a
regression head.  It preserves the original tagger API while
providing a clean classical baseline that can be extended to
quantum variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fraud‑Detection layer (classical)
@dataclass
class FraudLayerParameters:
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
            return outputs * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# Convolutional filter (classical)
def Conv() -> nn.Module:
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()

# Estimator QNN (classical regression head)
def EstimatorQNN() -> nn.Module:
    """Return a simple fully‑connected regression network."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()

# Classical LSTM cell
class ClassicalQLSTM(nn.Module):
    """Drop‑in replacement using classical linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        return torch.cat(outputs, dim=0), (hx, cx)

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

# Hybrid model
class HybridQLSTM(nn.Module):
    """Sequence tagging model that can mix classical and quantum LSTM layers
    and optionally apply fraud‑detection, convolution and regression heads.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_fraud: bool = False,
        use_conv: bool = False,
        use_estimator: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Use classical LSTM for all modes
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Optional components
        self.use_fraud = use_fraud
        self.fraud_preprocessor = None
        if use_fraud:
            dummy_params = FraudLayerParameters(
                bs_theta=0.5, bs_phi=0.5, phases=(0.1, 0.2),
                squeeze_r=(0.3, 0.3), squeeze_phi=(0.1, 0.1),
                displacement_r=(0.2, 0.2), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            self.fraud_preprocessor = build_fraud_detection_program(dummy_params, [])

        self.use_conv = use_conv
        self.conv_filter = None
        if use_conv:
            self.conv_filter = Conv()

        self.use_estimator = use_estimator
        self.estimator_head = None
        if use_estimator:
            self.estimator_head = EstimatorQNN()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Optional fraud pre‑processing
        if self.use_fraud and self.fraud_preprocessor is not None:
            flat = self.word_embeddings(sentence).view(-1, 2)
            processed = self.fraud_preprocessor(flat)
            processed = processed.view(sentence.size(0), -1, self.hidden_dim)
            lstm_input = processed
        else:
            lstm_input = self.word_embeddings(sentence).view(sentence.size(0), 1, -1)

        lstm_out, _ = self.lstm(lstm_input)
        tag_logits = self.hidden2tag(lstm_out.view(sentence.size(0), -1))

        # Optional estimator head
        if self.use_estimator and self.estimator_head is not None:
            est_out = self.estimator_head(lstm_out.view(-1, self.hidden_dim))
            est_out = est_out.view(sentence.size(0), -1)
            tag_logits = torch.cat([tag_logits, est_out], dim=-1)

        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM"]
