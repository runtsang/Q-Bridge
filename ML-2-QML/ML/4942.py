from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import torch
from torch import nn

# Import helper modules from the seed repository
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program
from.Autoencoder import Autoencoder, AutoencoderConfig
from.Conv import Conv
from.SamplerQNN import SamplerQNN

@dataclass
class FraudDetectionHybridConfig:
    fraud_input: FraudLayerParameters
    fraud_layers: Sequence[FraudLayerParameters]
    autoencoder_config: AutoencoderConfig
    conv_kernel: int = 2
    conv_threshold: float = 0.0

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model that chains a photonic‑style feature extractor,
    a classical auto‑encoder, a convolutional filter, and a sampler neural network.
    The architecture mirrors the quantum workflow while remaining fully classical."""
    def __init__(self, cfg: FraudDetectionHybridConfig) -> None:
        super().__init__()
        self.fraud_net = build_fraud_detection_program(cfg.fraud_input, cfg.fraud_layers)
        self.autoencoder = Autoencoder(
            input_dim=cfg.autoencoder_config.input_dim,
            latent_dim=cfg.autoencoder_config.latent_dim,
            hidden_dims=cfg.autoencoder_config.hidden_dims,
            dropout=cfg.autoencoder_config.dropout,
        )
        self.conv = Conv()
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Photonic‑style linear layers
        h = self.fraud_net(x)
        # 2. Encode‑decode autoencoder
        latent = self.autoencoder.encode(h)
        recon = self.autoencoder.decode(latent)
        # 3. Convolutional filter on a fabricated 2×2 patch
        batch = h.shape[0]
        patch = torch.stack([h[:,0], h[:,1], h[:,0], h[:,1]], dim=1).view(batch, 2, 2)
        conv_out = torch.stack([self.conv.run(p.cpu().numpy()) for p in patch])
        conv_out = conv_out.to(x.device).unsqueeze(-1)
        # 4. Sampler network expects a 2‑dimensional input; use one feature of h
        inp = torch.cat([h[:,0:1], conv_out], dim=-1)
        return self.sampler(inp)

__all__ = ["FraudDetectionHybrid", "FraudDetectionHybridConfig"]
