"""Hybrid estimator combining CNN features with a quantum layer.

The model is compatible with the legacy EstimatorQNN interface,
but now includes a deeper CNN encoder and a parameterized
four‑qubit quantum circuit.  It can be used as a drop‑in
replacement for the original feed‑forward regressor while
leveraging quantum correlations for richer feature
representations.
"""

import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QLayer(tq.QuantumModule):
    """Quantum feature layer inspired by Quantum‑NAT."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.crx(qdev, wires=[0, 3])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])


class EstimatorQNNHybrid(nn.Module):
    """CNN + quantum hybrid regressor."""

    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

        # Quantum encoder and layer
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.quantum_norm = nn.BatchNorm1d(self.q_layer.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        pooled = torch.nn.functional.avg_pool2d(feat, 6).view(bsz, 16)
        out = self.fc(pooled)
        out = self.norm(out)

        # Quantum processing
        qdev = tq.QuantumDevice(
            n_wires=self.q_layer.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        self.encoder(qdev, out)
        self.q_layer(qdev)
        q_out = self.measure(qdev)
        q_out = self.quantum_norm(q_out)
        return out + q_out  # simple ensemble


def EstimatorQNN() -> EstimatorQNNHybrid:
    """Legacy factory returning the hybrid estimator."""
    return EstimatorQNNHybrid()


class FastEstimator:
    """Fast deterministic or shot‑noise aware evaluator for EstimatorQNNHybrid."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "EstimatorQNNHybrid",
    "EstimatorQNN",
    "FastEstimator",
]
