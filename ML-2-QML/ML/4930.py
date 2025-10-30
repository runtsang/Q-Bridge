from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Callable

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. Shared parameter container
# ────────────────────────────────────────────────────
@dataclass
class FraudLayerParameters:
    """Unified description of a fraud‑detection layer for both classical and quantum circuits."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# ──────────────────────────────────────────────────────────────────────────────
# 2. Classical building blocks
# ──────────────────────────────────────────────────────────────────────────────
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_classical_fraud_model(input_params: FraudLayerParameters,
                               layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    """Construct a deep feed‑forward network that mirrors the photonic layers."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    # Add dropout and residual skip to improve robustness on large datasets
    net = nn.Sequential(
        *modules,
        nn.Dropout(p=0.1),
        nn.Linear(1, 1),
    )
    return net

# ──────────────────────────────────────────────────────────────────────────────
# 3. QCNN inspired feature extractor
# ──────────────────────────────────────────────────────────────────────────────
class QCNNModel(nn.Module):
    """A lightweight fully‑connected model that emulates the QCNN structure."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# ──────────────────────────────────────────────────────────────────────────────
# 4. Estimator utilities
# ──────────────────────────────────────────────────────────────────────────────
class FastBaseEstimator:
    """Evaluate a PyTorch model on a batch of parameter sets."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    row.append(float(val.mean().item()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to deterministic outputs."""
    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

# ──────────────────────────────────────────────────────────────────────────────
# 5. Sampler neural network
# ──────────────────────────────────────────────────────────────────────────────
def SamplerQNN() -> nn.Module:
    """A classical soft‑max classifier that can be swapped with a quantum SamplerQNN."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(x), dim=-1)

    return SamplerModule()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Hybrid wrapper
# ──────────────────────────────────────────────────────────────────────────────
class FraudDetectionHybrid:
    """Combines classical and quantum fraud‑detection models."""
    def __init__(self,
                 classical_params: Iterable[FraudLayerParameters],
                 quantum_params: Iterable[FraudLayerParameters],
                 use_qml: bool = True) -> None:
        self.classical = build_classical_fraud_model(classical_params[0], classical_params[1:])
        self.qml = None
        if use_qml:
            # Lazy import to avoid heavy dependencies until needed
            from.qml_code import build_quantum_fraud_program  # type: ignore
            self.qml = build_quantum_fraud_program(quantum_params[0], quantum_params[1:])

    def evaluate(self, params: Sequence[float]) -> float:
        """Return classical output; if quantum circuit present, average with quantum expectation."""
        class_out = self.classical(torch.as_tensor(params, dtype=torch.float32).unsqueeze(0))
        val = float(class_out.mean().item())
        if self.qml is None:
            return val
        # Quantum evaluation via StrawberryFields simulator
        from strawberryfields import Engine
        eng = Engine("fock", backend_options={"cutoff_dim": 8})
        eng.run(self.qml, args=dict(zip(self.qml.parameters, params)))
        state = eng.state
        exp_val = state.expectation_value("Z0")
        return (val + exp_val) / 2.0

__all__ = [
    "FraudLayerParameters",
    "build_classical_fraud_model",
    "QCNNModel",
    "FastBaseEstimator",
    "FastEstimator",
    "SamplerQNN",
    "FraudDetectionHybrid",
]
