"""
HybridQLSTM – classical implementation of the quantum‑enhanced LSTM, quantum convolution,
quantum classifier, and fraud‑detection pipeline.

All sub‑components are pure PyTorch modules and functions, providing an identical API
to the quantum version while remaining fully differentiable and GPU‑friendly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Classical Convolutional filter (drop‑in replacement for quanvolution)
# --------------------------------------------------------------------------- #
class Conv(nn.Module):
    """2‑D convolutional filter with a learnable bias and thresholding."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: Tensor of shape (..., kernel_size, kernel_size)
        Returns:
            Tensor of mean activation value after sigmoid thresholding.
        """
        x = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2, 3))


# --------------------------------------------------------------------------- #
# Classical quantum‑style LSTM (pure linear gates)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in classical equivalent of the quantum LSTM."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
# Classical classifier (feed‑forward with ReLU)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Builds a simple fully‑connected classifier that mimics the structure of the quantum
    variational circuit.  Returns the network and metadata for reproducibility.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# Classical fraud‑detection module
# --------------------------------------------------------------------------- #
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


def _layer_from_params(
    params: FraudLayerParameters, *, clip: bool
) -> nn.Module:
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
    layers: list[FraudLayerParameters],
) -> nn.Sequential:
    """
    Creates a sequential PyTorch model that mirrors the photonic fraud‑detection
    architecture.  The first layer is un‑clipped; subsequent layers are clipped
    to emulate the physical bounds of the photonic gates.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Hybrid model – classical implementation
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    A fully classical analogue of the quantum hybrid architecture.
    The module accepts a token sequence and a feature vector, processes them
    through a convolution, an LSTM, a classifier, and a fraud‑detection head.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_features: int,
        depth: int,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: list[FraudLayerParameters],
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional filter (classical)
        self.conv = Conv(kernel_size=2, threshold=0.0)

        # LSTM – classical or quantum style depending on ``n_qubits``
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)

        # Classifier network
        self.classifier, self.enc, self.wsize, self.obs = build_classifier_circuit(num_features, depth)

        # Fraud‑detection head
        self.fraud_net = build_fraud_detection_program(fraud_input_params, fraud_layers)

    def forward(self, sentence: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len, batch)
            features: FloatTensor of shape (batch, num_features)
        Returns:
            Tensor of shape (batch, 1) – fraud‑detection logits.
        """
        # 1. Embedding
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)

        # 2. LSTM
        lstm_out, _ = self.lstm(embeds)  # (seq_len, batch, hidden)

        # 3. Classifier (use the last hidden state as feature)
        feats = lstm_out[-1]  # (batch, hidden)
        cls_logits = self.classifier(feats)  # (batch, 2)

        # 4. Fraud detection
        fraud_logits = self.fraud_net(cls_logits)  # (batch, 1)
        return torch.sigmoid(fraud_logits)


__all__ = ["HybridQLSTM"]
