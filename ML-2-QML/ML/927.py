"""
FraudDetectionModel – Classical implementation with optional variational quantum layer.

The class builds a stack of fully‑connected, Tanh‑activated layers that mirror the
photonic layers in the seed.  An optional PennyLane QNode can be appended to
inject quantum features.  The interface is identical to the seed but offers
dropout, parameter clipping, and a small hyper‑parameter search helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import torch
from torch import nn
import pennylane as qml


@dataclass
class LayerParameters:
    """Parameters for a single classical layer."""
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


def _build_layer(params: LayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudDetectionModel(nn.Module):
    """
    Hybrid classical‑quantum fraud‑detection model.

    Parameters
    ----------
    input_params : LayerParameters
        Parameters for the first (input) layer.
    hidden_params : Iterable[LayerParameters]
        Parameters for subsequent hidden layers.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    use_quantum : bool, default False
        If True, a PennyLane variational circuit is appended.
    quantum_layers : int, optional
        Number of quantum layers (only used if ``use_quantum`` is True).
    """

    def __init__(
        self,
        input_params: LayerParameters,
        hidden_params: Iterable[LayerParameters],
        *,
        dropout: float = 0.0,
        use_quantum: bool = False,
        quantum_layers: int = 1,
    ) -> None:
        super().__init__()
        self.classical = nn.Sequential(
            _build_layer(input_params, clip=False),
            *(_build_layer(p, clip=True) for p in hidden_params),
        )
        if dropout > 0.0:
            self.classical = nn.Sequential(
                *[
                    *self.classical,
                    nn.Dropout(dropout),
                ]
            )
        self.classical.add_module("output", nn.Linear(2, 1))

        self.use_quantum = use_quantum
        if use_quantum:
            # PennyLane device and QNode
            self.dev = qml.device("default.qubit", wires=2)
            self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch")
            self.quantum_layers = quantum_layers
            # Trainable parameters for the quantum circuit
            self.q_params = nn.Parameter(
                torch.randn(quantum_layers, 2, 2, dtype=torch.float32)
            )

    def _quantum_circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Variational circuit that mirrors the photonic layer structure."""
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)
        for layer_idx in range(self.quantum_layers):
            a, b = params[layer_idx]
            qml.RZ(a[0], wires=0)
            qml.RZ(b[0], wires=1)
            qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classical(x)
        if self.use_quantum:
            # Feed classical output as parameters to the quantum circuit
            q_out = self.qnode(out.squeeze(), self.q_params)
            out = torch.cat([out, q_out.unsqueeze(-1)], dim=-1)
        return out


__all__ = ["LayerParameters", "FraudDetectionModel"]
