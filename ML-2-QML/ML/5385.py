import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Callable, List
from dataclasses import dataclass

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

def _layer_from_params(params: FraudLayerParameters, clip: bool = False) -> nn.Linear:
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
    return linear

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QCNNModel(nn.Module):
    """Classical approximation of the QCNN feature extractor."""
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class SelfAttentionHybrid(nn.Module):
    """Hybrid self‑attention model that combines a classical attention block,
    a QCNN‑style feature extractor and fraud‑inspired weight clipping.
    """
    def __init__(self,
                 embed_dim: int,
                 rotation_params: np.ndarray,
                 entangle_params: np.ndarray,
                 qcnn_layers: int = 2,
                 fraud_params: FraudLayerParameters | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Store rotation and entangle matrices as trainable parameters
        self.rotation = nn.Parameter(
            torch.as_tensor(rotation_params.reshape(embed_dim, -1), dtype=torch.float32)
        )
        self.entangle = nn.Parameter(
            torch.as_tensor(entangle_params.reshape(embed_dim, -1), dtype=torch.float32)
        )
        # Optional QCNN feature extractor
        self.qcnn = QCNNModel() if qcnn_layers > 0 else None
        # Optional fraud‑style linear layer for bias initialization
        if fraud_params:
            self.bias_layer = _layer_from_params(fraud_params, clip=True)
        else:
            self.bias_layer = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Classical self‑attention (same as in SelfAttention.py)
        query = torch.matmul(inputs, self.rotation)
        key = torch.matmul(inputs, self.entangle)
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ value
        # Optional QCNN feature extraction
        if self.qcnn:
            attn_out = self.qcnn(attn_out)
        # Optional fraud bias layer
        attn_out = self.bias_layer(attn_out)
        return attn_out

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        """Evaluate the model for many input parameter sets.
        Parameters are expected to be the raw input vectors for the attention block.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            rng = np.random.default_rng(seed)
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                if shots is not None:
                    # Gaussian shot noise
                    row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                results.append(row)
        return results

def SelfAttention() -> SelfAttentionHybrid:
    """Factory that returns a ready‑to‑use hybrid self‑attention model."""
    embed_dim = 4
    rotation_params = np.random.randn(embed_dim * 3)
    entangle_params = np.random.randn(embed_dim * 3)
    fraud_params = FraudLayerParameters(
        bs_theta=0.5, bs_phi=0.3,
        phases=(0.1, -0.2),
        squeeze_r=(0.4, 0.6),
        squeeze_phi=(0.0, 0.0),
        displacement_r=(0.2, 0.3),
        displacement_phi=(0.1, -0.1),
        kerr=(0.0, 0.0)
    )
    return SelfAttentionHybrid(
        embed_dim=embed_dim,
        rotation_params=rotation_params,
        entangle_params=entangle_params,
        qcnn_layers=2,
        fraud_params=fraud_params
    )

__all__ = ["SelfAttentionHybrid", "SelfAttention"]
