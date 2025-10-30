from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, List

# ----------------------------------------------------------------------
# 1. Classical Autoencoder
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight MLP autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    """Convenience factory mirroring the quantum helper."""
    return AutoencoderNet(cfg)

# ----------------------------------------------------------------------
# 2. Classical Quanvolution Filter
# ----------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """2x2 patch convolution with stride 2."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)
        return feats.view(x.size(0), -1)

# ----------------------------------------------------------------------
# 3. Photonic Fraud‑Detection Style Layer
# ----------------------------------------------------------------------
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

def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            return out * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    """Build a sequential fraud‑detection model."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# 4. Classical Classifier Builder (Depth‑controlled)
# ----------------------------------------------------------------------
def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.extend([lin, nn.ReLU()])
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# ----------------------------------------------------------------------
# 5. Main Hybrid Model
# ----------------------------------------------------------------------
class QuanvolutionHybridNet(nn.Module):
    """
    Classical pipeline:
    1. Autoencoder (latent bottleneck)
    2. Classical quanvolution filter
    3. Fraud‑detection style layer (optional)
    4. Classifier built from a depth‑controlled feed‑forward network
    """
    def __init__(self,
                 autoencoder_cfg: AutoencoderConfig,
                 classifier_depth: int = 2,
                 fraud_layers: Iterable[FraudLayerParameters] = ()) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(autoencoder_cfg)
        self.qfilter = QuanvolutionFilter()
        self.fraud = build_fraud_detection_program(
            FraudLayerParameters(0.0, 0.0, (0.0, 0.0), (0.0, 0.0),
                                 (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            fraud_layers)
        # Determine feature size after quanvolution
        dummy = torch.zeros(1, 1, 28, 28)
        feat = self.qfilter(dummy)
        feat_size = feat.shape[1]
        self.classifier, _, _, _ = build_classifier_circuit(feat_size, classifier_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 1, 28, 28]
        Returns:
            logits: log‑softmax over two classes
            fraud_out: fraud‑detection layer output (shape [batch, 1])
        """
        # Flatten for autoencoder
        flat = x.view(x.size(0), -1)
        latent = self.autoencoder.encode(flat)
        recon = self.autoencoder.decode(latent).view(x.shape)
        qfeat = self.qfilter(recon)          # [batch, feat_size]
        fraud_out = self.fraud(qfeat[:, :2])  # use first two dims
        logits = self.classifier(qfeat)
        return F.log_softmax(logits, dim=-1), fraud_out

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "QuanvolutionFilter",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "QuanvolutionHybridNet",
]
