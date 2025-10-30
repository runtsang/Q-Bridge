"""Hybrid fraud detection model using classical photonic layers and optional quantum LSTM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional torchquantum import; if unavailable, fallback to classical LSTM
try:
    import torchquantum as tq
except Exception:  # pragma: no cover
    tq = None  # type: ignore[assignment]

# --- Parameters for photonic layers ------------------------------------
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

# --- Utility functions --------------------------------------------------
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
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
            return outputs * self.scale + self.shift
    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --- Quantum LSTM wrapper (fallback to classical if torchquantum not present) ----
class _QLSTMCell(nn.Module):
    """Gate‑based LSTM cell implemented with torchquantum. Falls back to a linear map."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        if tq is not None:
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        else:
            # Simple linear transformation as a placeholder
            self.projection = nn.Linear(n_wires, n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tq is None:
            return self.projection(x)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        for i in range(self.n_wires - 1):
            tq.cnot(qdev, wires=[i, i + 1])
        tq.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

# --- FraudDetectionHybrid ------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection pipeline that can operate in either
    a purely classical or a quantum‑enhanced mode.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        n_qubits: int = 0,
        hidden_dim: int = 64,
        tagset_size: int = 5,
    ) -> None:
        super().__init__()
        self.use_quantum = n_qubits > 0
        # Head: fraud‑detection layers
        self.head = build_fraud_detection_program(input_params, layers)
        # Sequence encoder
        if self.use_quantum:
            self.encoder = _QLSTMCell(n_qubits)
        else:
            self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feat)
        if isinstance(self.encoder, nn.LSTM):
            out, _ = self.encoder(x)
        else:
            out, _ = self.encoder(x)
        logits = self.output(out)
        return F.log_softmax(logits, dim=-1)

    # --- Evaluation helpers ---------------------------------------------
    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model over multiple parameter sets.
        For classical models, a simple shot‑noise wrapper is applied.
        """
        if observables is None:
            observables = [lambda out: out.mean()]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shots is not None:
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(torch.randn(1, generator=rng).item() *
                                   max(1e-6, 1 / shots) + mean) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
