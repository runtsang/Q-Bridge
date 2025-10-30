"""Combined classical and quantum hybrid model with fraud detection and estimator heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Sequence, Callable, List, Any

# Classical Quanvolution filter
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

# Fraud detection components
class FraudLayerParameters:
    def __init__(self, bs_theta: float, bs_phi: float, phases: tuple[float, float],
                 squeeze_r: tuple[float, float], squeeze_phi: tuple[float, float],
                 displacement_r: tuple[float, float], displacement_phi: tuple[float, float],
                 kerr: tuple[float, float]):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
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

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# EstimatorQNN regressor
def EstimatorQNN() -> nn.Module:
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)
    return EstimatorNN()

# Fast estimator utilities
class FastBaseEstimator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(self,
                 observables,
                 parameter_sets,
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# Shared hybrid class
class QuanvolutionHybrid(nn.Module):
    """Hybrid model that can operate in classical mode with optional fraud-detection or estimator head."""
    def __init__(self,
                 mode: str = "classical",
                 head: str = "linear",
                 fraud_params: Sequence[FraudLayerParameters] | None = None,
                 estimator: bool = False) -> None:
        super().__init__()
        if mode!= "classical":
            raise NotImplementedError("Only classical mode is implemented in this module.")
        self.filter = QuanvolutionFilter()
        if head == "linear":
            self.head = nn.Linear(4 * 14 * 14, 10)
        elif head == "fraud":
            if not fraud_params:
                raise ValueError("fraud_params required for fraud head")
            self.reduction = nn.Linear(4 * 14 * 14, 2)
            self.head = build_fraud_detection_program(fraud_params[0], fraud_params[1:])
        elif head == "estimator":
            self.reduction = nn.Linear(4 * 14 * 14, 2)
            self.head = EstimatorQNN()
        else:
            raise ValueError(f"Unsupported head {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        if hasattr(self, "reduction"):
            features = self.reduction(features)
        logits = self.head(features)
        # For linear head we apply log_softmax; for others we return raw logits
        if isinstance(self.head, nn.Linear):
            return F.log_softmax(logits, dim=-1)
        else:
            return logits

__all__ = ["QuanvolutionHybrid",
           "FastBaseEstimator",
           "FastEstimator",
           "FraudLayerParameters",
           "EstimatorQNN"]
