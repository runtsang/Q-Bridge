import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

# ------------------------------------------------------------------
# 1. Classical convolutional filter (from Conv.py)
# ------------------------------------------------------------------
class ConvFilter(nn.Module):
    """
    A lightweight 2‑D convolutional filter that mimics the behaviour of a quanvolution layer.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """
        Forward pass that returns the mean activation after sigmoid thresholding.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# ------------------------------------------------------------------
# 2. Fraud‑detection style layer (from FraudDetection.py)
# ------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
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


# ------------------------------------------------------------------
# 3. Classical self‑attention (from SelfAttention.py)
# ------------------------------------------------------------------
class ClassicalSelfAttention:
    """
    Simple dot‑product self‑attention implemented in PyTorch.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# ------------------------------------------------------------------
# 4. Auto‑encoder (from Autoencoder.py)
# ------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


# ------------------------------------------------------------------
# 5. Hybrid pipeline
# ------------------------------------------------------------------
class HybridSelfAttentionModel:
    """
    A hybrid model that chains a convolutional filter, a fraud‑detection layer,
    a self‑attention block, and an auto‑encoder.  The same class name is
    re‑used in the quantum module for a direct mapping.
    """
    def __init__(
        self,
        conv_kernel: int = 2,
        fraud_params: Optional[FraudLayerParameters] = None,
        attention_dim: int = 4,
        autoencoder_config: Optional[AutoencoderConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.conv = ConvFilter(kernel_size=conv_kernel)
        default_fraud = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud_layer = _layer_from_params(
            fraud_params or default_fraud, clip=True
        )
        self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
        self.autoencoder = AutoencoderNet(
            autoencoder_config or AutoencoderConfig(input_dim=attention_dim)
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass that follows the hybrid pipeline and returns a NumPy array.
        """
        # 1. Convolutional filtering
        conv_out = self.conv.run(inputs)

        # 2. Fraud‑detection style feature extraction
        #   The fraud layer expects a 2‑D tensor; we feed the conv output as a 2‑element vector.
        fraud_input = torch.tensor(
            [[conv_out, conv_out]], dtype=torch.float32, device=self.device
        )
        fraud_out = self.fraud_layer(fraud_input)

        # 3. Self‑attention
        #   For simplicity we reuse the fraud output as both rotation and entangle params.
        rotation_params = np.asarray(fraud_out.cpu().numpy()).flatten()
        entangle_params = rotation_params.copy()
        attention_out = self.attention.run(rotation_params, entangle_params, conv_out)

        # 4. Auto‑encoder
        attention_tensor = torch.tensor(
            [attention_out], dtype=torch.float32, device=self.device
        )
        latent = self.autoencoder.encode(attention_tensor)
        reconstruction = self.autoencoder.decode(latent)

        return reconstruction.cpu().numpy()

    def run_classical(self, inputs: np.ndarray) -> np.ndarray:
        """
        Public entry‑point that accepts raw NumPy data.
        """
        return self.forward(inputs)

__all__ = ["HybridSelfAttentionModel"]
