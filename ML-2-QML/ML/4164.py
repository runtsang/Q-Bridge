"""FraudDetectionModel – classical implementation.

This module builds a PyTorch model that mirrors the photonic fraud‑detection
circuit, optionally injects a classical LSTM (or its quantum‑inspired counterpart),
and provides a classical analogue of a quantum fully‑connected layer.

The design is intentionally modular so that each component can be swapped
for a quantum version in the QML module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


# -------------------- 1. Layer definition ------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a value to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single PyTorch layer from photonic parameters."""
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a sequential PyTorch model that mimics the photonic stack."""
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# -------------------- 2. Classical LSTM (drop‑in replacement) ----------------
class ClassicalQLSTM(nn.Module):
    """Classical LSTM that mirrors the interface of the quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# -------------------- 3. Classical FCL analogue -----------------------------
def FCL() -> nn.Module:
    """Return a PyTorch module that mimics the quantum FCL example."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().cpu().numpy()

    return FullyConnectedLayer()


# -------------------- 4. High‑level FraudDetectionModel --------------------
class FraudDetectionModel(nn.Module):
    """
    Hybrid model that can be configured to use:
      * photonic‑style layers (classical)
      * optional classical LSTM (QLSTM)
      * optional fully‑connected quantum layer analogue
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_layers: Iterable[FraudLayerParameters],
        n_qubits: int = 0,
        use_fcl: bool = False,
    ) -> None:
        super().__init__()
        self.main = build_fraud_detection_program(input_params, hidden_layers)
        self.lstm = ClassicalQLSTM(2, 2, n_qubits) if n_qubits > 0 else None
        self.fcl = FCL() if use_fcl else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        if self.lstm is not None:
            out, _ = self.lstm(out.unsqueeze(0))
            out = out.squeeze(0)
        if self.fcl is not None:
            out = torch.from_numpy(self.fcl.run(out.detach().cpu().numpy()))
        return out

__all__ = [
    "FraudLayerParameters",
    "_layer_from_params",
    "build_fraud_detection_program",
    "ClassicalQLSTM",
    "FCL",
    "FraudDetectionModel",
]
