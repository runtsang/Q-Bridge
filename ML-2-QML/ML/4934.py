from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple, Callable
import torch
from torch import nn
import torch.nn.functional as F


class FraudDetectionHybrid:
    """Hybrid classical fraud‑detection model with optional Quanvolution branch."""

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

    def _build_layer(
        self,
        params: "FraudDetectionHybrid.FraudLayerParameters",
        clip: bool,
    ) -> nn.Module:
        """Create a single photonic‑inspired layer with optional clipping."""
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

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.activation(self.linear(x))
                return out * self.scale + self.shift

        return Layer()

    def build_classical_fraud_model(
        self,
        input_params: "FraudDetectionHybrid.FraudLayerParameters",
        layers: Iterable["FraudDetectionHybrid.FraudLayerParameters"],
        image_shape: Tuple[int, int, int] = (1, 28, 28),
    ) -> nn.Module:
        """
        Assemble a hybrid network that processes a 2‑D feature vector
        with photonic‑inspired layers and extracts spatial patterns via
        a Quanvolution filter.
        """
        seq = nn.ModuleList()
        seq.append(self._build_layer(input_params, clip=False))
        seq.extend(self._build_layer(l, clip=True) for l in layers)
        seq.append(nn.Linear(2, 1))

        # Quanvolution branch
        qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        flatten_size = (image_shape[1] // 2 - 1) * (image_shape[2] // 2 - 1) * 4
        linear_head = nn.Linear(flatten_size, 10)

        class HybridNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fraud_seq = nn.Sequential(*seq)
                self.qfilter = qfilter
                self.linear_head = linear_head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Expect first two columns as the 2‑D feature vector
                vec = x[:, :2]
                out_vec = self.fraud_seq(vec)
                img = x[:, 2:].reshape(-1, *image_shape)
                feats = self.qfilter(img)
                feats = feats.view(img.size(0), -1)
                logits = self.linear_head(feats)
                return F.log_softmax(torch.cat([out_vec, logits], dim=-1), dim=-1)

        return HybridNet()

    class FastEstimator:
        """Evaluator that supports optional shot noise for the classical network."""

        def __init__(self, model: nn.Module) -> None:
            self.model = model

        def evaluate(
            self,
            observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
            parameter_sets: Sequence[Sequence[float]],
            *,
            shots: int | None = None,
            seed: int | None = None,
        ) -> List[List[float]]:
            observables = list(observables) or [lambda out: out.mean(dim=-1)]
            results: List[List[float]] = []
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inp = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                    out = self.model(inp)
                    row = []
                    for obs in observables:
                        val = obs(out)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().item())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)
            if shots is None:
                return results
            rng = torch.Generator().manual_seed(seed or 0)
            noisy = []
            for row in results:
                noisy_row = [
                    float(torch.randn(1, generator=rng).item() / shots + r) for r in row
                ]
                noisy.append(noisy_row)
            return noisy

    __all__ = ["FraudDetectionHybrid"]
