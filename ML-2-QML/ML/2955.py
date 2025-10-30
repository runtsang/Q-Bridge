"""FraudDetectionHybridNet – Classical‑Quantum hybrid for fraud classification.

The implementation merges:
• 1️⃣ Classical convolutional backbone (from ClassicalQuantumBinaryClassification.py) that extracts image‑style features.
• 2️⃣ Photonic‑layer inspired feature map (from FraudDetection.py) that is applied to the 2‑mode output of the CNN.
• 3️⃣ A differentiable hybrid layer that forwards the activations through a Qiskit or Strawberry‑Fields circuit and returns an expectation value.
The class is fully importable and can be used as a torch.nn.Module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Callable

import torch
from torch import nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. Photonic‑layer parameters and helper
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

# ------------------------------------------------------------------
# 2. Photonic program builder (kept for reference)
# ------------------------------------------------------------------
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> "sf.Program":
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

# ------------------------------------------------------------------
# 3. Classical convolutional backbone
# ------------------------------------------------------------------
class _ConvBackbone(nn.Module):
    """Simple CNN that mirrors the structure of QCNet in ClassicalQuantumBinaryClassification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # The flattened size depends on the input image; we assume 32x32 for this example.
        self.fc1 = nn.Linear(15 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        return x

# ------------------------------------------------------------------
# 4. Hybrid fraud‑detection head
# ------------------------------------------------------------------
class FraudDetectionHybridNet(nn.Module):
    """
    End‑to‑end hybrid model for fraud detection.

    Parameters
    ----------
    hybrid_layer : nn.Module
        A module that accepts a 2‑dimensional torch tensor and returns a 1‑dimensional
        expectation value.  The layer should be differentiable via PyTorch's autograd.
    """
    def __init__(self, hybrid_layer: nn.Module) -> None:
        super().__init__()
        self.backbone = _ConvBackbone()
        self.fc3 = nn.Linear(84, 2)          # produce 2‑mode feature vector
        self.hybrid = hybrid_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc3(x)                      # shape (..., 2)
        # hybrid layer expects shape (batch, 2)
        logits = self.hybrid(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "FraudLayerParameters",
    "_layer_from_params",
    "build_fraud_detection_program",
    "FraudDetectionHybridNet",
]
