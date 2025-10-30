from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# Classical Estimators
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: list[callable],
        parameter_sets: list[list[float]],
    ) -> list[list[float]]:
        if not observables:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        results: list[list[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: list[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().item())
                    row.append(val)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic outputs."""
    def evaluate(
        self,
        observables: list[callable],
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: list[list[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# ----------------------------------------------------------------------
# Classical Quanvolution Components
# ----------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """Apply a 2×2 convolution to extract patches."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses a quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


# ----------------------------------------------------------------------
# Classical QCNN Model
# ----------------------------------------------------------------------
class QCNNModel(nn.Module):
    """
    Hybrid classical QCNN that optionally uses quanvolution patches
    and an optional LSTM for sequential feature refinement.
    """
    def __init__(
        self,
        *,
        use_quanvolution: bool = False,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.use_lstm = use_lstm

        if use_quanvolution:
            self.qfilter = QuanvolutionFilter()
            feature_dim = 4 * 14 * 14
        else:
            feature_dim = 8  # placeholder for raw flattened input

        self.conv1 = nn.Linear(feature_dim, 16)
        self.conv2 = nn.Linear(16, 8)
        self.pool = nn.Linear(8, 4)

        if use_lstm:
            self.lstm = nn.LSTM(4, lstm_hidden_dim, batch_first=True)
            self.classifier = nn.Linear(lstm_hidden_dim, 1)
        else:
            self.classifier = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.qfilter(x)
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.pool(x))
        if self.use_lstm:
            x, _ = self.lstm(x.unsqueeze(1))
            x = x.squeeze(1)
        return torch.sigmoid(self.classifier(x))


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def QCNN(
    *,
    use_quanvolution: bool = False,
    use_lstm: bool = False,
    lstm_hidden_dim: int = 8,
    noise_shots: int | None = None,
    seed: int | None = None,
) -> nn.Module:
    """
    Construct a classical QCNN model, optionally wrapped in a FastEstimator
    to inject shot noise.  Parameters mirror the seed QCNN but expose
    quanvolution and LSTM switches.
    """
    model = QCNNModel(
        use_quanvolution=use_quanvolution,
        use_lstm=use_lstm,
        lstm_hidden_dim=lstm_hidden_dim,
    )
    if noise_shots is None:
        return model
    return FastEstimator(model)


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "QCNNModel",
    "QCNN",
]
