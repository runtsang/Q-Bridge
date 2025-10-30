import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable

# Classical convolutional filter
class ClassicalConv(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data expected shape (batch, 1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3], keepdim=True)

# Classical self‑attention
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

# Fraud‑detection network
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# Classical LSTM
class ClassicalQLSTM(nn.Module):
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
        states: Iterable[torch.Tensor] | None = None,
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
        states: Iterable[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# Hybrid model
class HybridQLSTM(nn.Module):
    """Combines embedding, classical convolution, self‑attention, LSTM and fraud‑detection
    layers into a single end‑to‑end tagger.  All sub‑modules are classical and
    fully differentiable, enabling standard gradient‑based optimisation."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_conv: bool = True,
        use_attention: bool = True,
        use_fraud: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.use_conv = use_conv
        self.use_attention = use_attention
        self.use_fraud = use_fraud

        if use_conv:
            self.conv = ClassicalConv(kernel_size=2, threshold=0.0)
        if use_attention:
            self.attention = ClassicalSelfAttention(embed_dim=embedding_dim)
        if use_fraud:
            fraud_params = FraudLayerParameters(
                bs_theta=0.5, bs_phi=0.5,
                phases=(0.1, 0.1),
                squeeze_r=(0.2, 0.2),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.3, 0.3),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            self.fraud_net = build_fraud_detection_program(
                fraud_params,
                layers=[fraud_params, fraud_params],
            )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        if self.use_conv:
            # dummy conv; output ignored in demo
            _ = self.conv(embeds.unsqueeze(1))
        if self.use_attention:
            rotation = np.random.randn(self.embedding.embedding_dim * 4)
            entangle = np.random.randn(self.embedding.embedding_dim * 4)
            embeds = self.attention(rotation, entangle, embeds)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        tag_logits = F.log_softmax(logits, dim=1)
        if self.use_fraud:
            fraud_out = self.fraud_net(lstm_out[-1].unsqueeze(0))
            tag_logits = torch.cat([tag_logits, fraud_out], dim=1)
        return tag_logits

__all__ = ["HybridQLSTM"]
