from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.quantum as tq

@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑inspired linear layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(val: float, bound: float) -> float:
    return max(-bound, min(bound, val))

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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a simple feed‑forward network that mirrors the photonic layer design."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QuantumPatchEncoder(tq.QuantumModule):
    """Random two‑qubit kernel applied to every 2×2 patch of an image."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2×2 patches, run the kernel, and return a flattened feature vector."""
        bsz = x.shape[0]
        device = x.device
        x_img = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x_img[:, r, c],
                        x_img[:, r, c + 1],
                        x_img[:, r + 1, c],
                        x_img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                meas = self.measure(qdev).view(bsz, 4)
                patches.append(meas)
        return torch.cat(patches, dim=1)

class FraudDetectionNet(nn.Module):
    """
    Hybrid model that fuses a classical photonic‑inspired MLP with a quantum patch encoder.
    The classical branch processes two hand‑crafted statistical features (mean and std)
    while the quantum branch extracts high‑dimensional quantum‑encoded features from image patches.
    The concatenated representation is classified by a final linear layer.
    """
    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        layer_params: List[FraudLayerParameters],
        encoder: QuantumPatchEncoder | None = None,
    ) -> None:
        super().__init__()
        self.classical_branch = build_fraud_detection_program(fraud_params, layer_params)
        self.encoder = encoder or QuantumPatchEncoder()
        self.classifier = nn.Linear(4 * 14 * 14 + 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28) grayscale image
        # Classical features: mean and std of pixel intensities
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = x.std(dim=[1, 2, 3], keepdim=True)
        classical_feat = torch.cat([mean, std], dim=1)
        classical_out = self.classical_branch(classical_feat)

        quantum_feat = self.encoder(x)
        combined = torch.cat([classical_out, quantum_feat], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuantumPatchEncoder", "FraudDetectionNet"]
