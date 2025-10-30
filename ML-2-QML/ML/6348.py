from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Tuple


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


class FraudDetectionHybrid(nn.Module):
    """
    Classical encoder that maps a 2‑D fraud feature vector to
    photonic parameters and a quantum‑sampler classifier.
    """

    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        self.num_layers = num_layers

        # Encoder: 2 → 2*num_layers*8 parameters
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, num_layers * 8 * 2),
            nn.Tanh(),
        )

        # Sampler head: 2 → 2 classes
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns class logits and a list of photonic layer parameters.
        """
        flat_params = self.encoder(x)
        params = self._unpack_params(flat_params)
        logits = self.sampler(x)
        return logits, params

    def _unpack_params(self, flat: torch.Tensor) -> List[FraudLayerParameters]:
        """Convert flattened tensor to a list of FraudLayerParameters."""
        params_list: List[FraudLayerParameters] = []
        per_layer = 8 * 2  # 8 parameters, each with 2 values
        for i in range(self.num_layers):
            start = i * per_layer
            chunk = flat[start : start + per_layer].view(8, 2)
            params_list.append(
                FraudLayerParameters(
                    bs_theta=chunk[0, 0].item(),
                    bs_phi=chunk[0, 1].item(),
                    phases=(chunk[1, 0].item(), chunk[1, 1].item()),
                    squeeze_r=(chunk[2, 0].item(), chunk[2, 1].item()),
                    squeeze_phi=(chunk[3, 0].item(), chunk[3, 1].item()),
                    displacement_r=(chunk[4, 0].item(), chunk[4, 1].item()),
                    displacement_phi=(chunk[5, 0].item(), chunk[5, 1].item()),
                    kerr=(chunk[6, 0].item(), chunk[6, 1].item()),
                )
            )
        return params_list

    def get_photonic_params(self, x: torch.Tensor) -> List[FraudLayerParameters]:
        """Convenience wrapper to obtain photonic parameters for a given input."""
        with torch.no_grad():
            flat = self.encoder(x)
        return self._unpack_params(flat)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
