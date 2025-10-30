import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterable, Sequence

# =========================
# Data generation
# =========================
def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic regression data by sampling points on a unit sphere
    and computing a non‑linear target.  The input is returned as a real
    tensor so that it can be fed directly into the classical model.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper for the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {"states": self.features[index], "target": self.labels[index]}

# =========================
# Fraud‑detection style layers
# =========================
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

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Build a sequential PyTorch model that mirrors the fraud‑detection photonic circuit.
    The first layer is unclipped; subsequent layers are clipped to keep parameters bounded.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# =========================
# Quantum‑inspired random feature layer
# =========================
class RandomFeatureLayer(nn.Module):
    """
    Simulate a quantum random layer using a fixed random projection followed
    by a cosine non‑linearity (a random Fourier feature map).
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.weight = nn.Parameter(torch.tensor(rng.standard_normal((in_features, out_features)), dtype=torch.float32),
                                   requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(rng.standard_normal(out_features), dtype=torch.float32),
                                 requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.weight + self.bias
        return torch.cos(z)

# =========================
# Classical patch extractor (quanvolution)
# =========================
class QuanvolutionFilter(nn.Module):
    """
    Extract 2×2 patches from a 28×28 image and flatten them into a feature vector.
    """
    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.stride = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (N, 1, 28, 28)
        patches = x.unfold(2, self.patch_size, self.stride) \
                     .unfold(3, self.patch_size, self.stride)
        # patches shape (N, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(x.size(0), -1, self.patch_size * self.patch_size)
        return patches.view(x.size(0), -1)

# =========================
# Hybrid regression model
# =========================
class HybridRegressionModel(nn.Module):
    """
    Hybrid regression model that combines:
      * A fraud‑style parameterized linear stack.
      * A quantum‑inspired random Fourier feature mapping.
      * A classical patch extractor for image data.
      * A final linear head for regression.
    """
    def __init__(self, num_features: int, fraud_layers: Iterable[FraudLayerParameters]):
        super().__init__()
        # Fraud detection style stack
        self.fraud_net = build_fraud_detection_program(fraud_layers[0], fraud_layers[1:])
        # Random feature mapping
        self.random_layer = RandomFeatureLayer(num_features, 64)
        # Classical patch extractor (if input is image)
        self.patch_extractor = QuanvolutionFilter()
        # Final regression head
        # 64 from random layer + 1 from fraud stack output
        self.head = nn.Linear(64 + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fraud part (uses the first two input dimensions)
        fraud_out = self.fraud_net(x[:, :2])  # shape (N, 1)
        # Random feature part
        rand_out = self.random_layer(x)  # shape (N, 64)
        # Patch extractor if input has image shape
        if x.ndim == 4:
            patch_out = self.patch_extractor(x)  # shape (N, 784)
            rand_out = torch.cat([rand_out, patch_out], dim=1)
        features = torch.cat([fraud_out, rand_out], dim=1)
        return self.head(features).squeeze(-1)

__all__ = [
    "RegressionDataset",
    "generate_superposition_data",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "RandomFeatureLayer",
    "QuanvolutionFilter",
    "HybridRegressionModel",
]
