"""Hybrid QCNN‑Fraud detection model with a shared feature‑map interface.

The module implements:
* A classical convolution‑like backbone (QCNN‑style fully‑connected layers).
* A photonic fraud‑detection block built from the same `FraudLayerParams` data structure.
* A wrapper that concatenates the feature embeddings from both branches before a final linear output.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# 1. Parameter container reused from the photonic seed
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParams:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Classical QCNN backbone (modified from seed)
# --------------------------------------------------------------------------- #
class QCNNBackbone(nn.Module):
    """Fully‑connected network that mimics the QCNN convolution‑pooling steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x

# --------------------------------------------------------------------------- #
# 3. Photonic fraud‑detection block (classical emulation)
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParams, *, clip: bool) -> nn.Module:
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
            out = out * self.scale + self.shift
            return out
    return Layer()

def build_fraud_detection_network(
    input_params: FraudLayerParams,
    layers: Iterable[FraudLayerParams],
) -> nn.Sequential:
    """Create a sequential network mirroring the photonic fraud‑detection layers."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 4. Hybrid model
# --------------------------------------------------------------------------- #
class QCNNFraudHybrid(nn.Module):
    """Hybrid QCNN‑Fraud detection model.

    Parameters
    ----------
    fraud_input : FraudLayerParams
        Parameters for the first photonic layer.
    fraud_layers : Iterable[FraudLayerParams]
        Parameters for subsequent photonic layers.
    """
    def __init__(
        self,
        fraud_input: FraudLayerParams,
        fraud_layers: Iterable[FraudLayerParams],
    ) -> None:
        super().__init__()
        self.backbone = QCNNBackbone()
        self.fraud_net = build_fraud_detection_network(fraud_input, fraud_layers)
        self.final = nn.Linear(5, 1)  # 4 from backbone + 1 from fraud output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone produces 4‑dim embedding
        xb = self.backbone(x)
        # fraud branch expects 2‑dim input; we reduce the feature vector
        xf = self.fraud_net(x[:, :2])  # use first two dims as a simple interface
        out = torch.cat([xb, xf], dim=1)
        return torch.sigmoid(self.final(out))

__all__ = ["QCNNFraudHybrid", "FraudLayerParams"]
