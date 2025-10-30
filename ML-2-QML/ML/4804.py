"""Unified hybrid layer combining classical dense, LSTM‑style gates, and photonic fraud‑detection logic.

The class exposes a single forward method that can be used as a drop‑in replacement for the original
FCL example.  It keeps the same public API (`FCL()`) while internally fusing three distinct
designs:

* A dense layer (from the original FCL)
* LSTM‑style gating realised with small quantum modules (from QLSTM)
* A photonic fraud‑detection transformation (from FraudDetection)

This gives users a single layer that can be trained classically while
leveraging quantum modules for the gating logic, and it also provides a
fairly cheap photonic style post‑processing that mimics the original
continuous‑variable circuit.

The layer is fully differentiable with PyTorch and can therefore be
plugged into any standard training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Fraud‑detection parameter container & helper
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters that describe a single photonic fraud layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _fraud_layer_from_params(params: FraudLayerParameters, *, clip: bool = False) -> nn.Module:
    """Create a classical fraud‑layer that mimics the photonic circuit.

    The module consists of a linear transform, a tanh activation, a scale
    and a shift.  This mirrors the behaviour of the Strawberry‑Fields
    program while remaining fully differentiable in PyTorch.
    """
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


# --------------------------------------------------------------------------- #
# 2. Quantum gate module (torchquantum)
# --------------------------------------------------------------------------- #

import torchquantum as tq
import torchquantum.functional as tqf

class QuantumGate(tq.QuantumModule):
    """Small variational circuit that implements one LSTM gate.

    It takes as input a vector of length ``n_wires`` and returns the
    expectation value of Pauli‑Z on all wires.  The circuit consists
    of input‑dependent RX rotations, a trainable RX on each wire,
    followed by a linear chain of CNOTs.  This design is inspired by
    the QLSTM implementation but written from scratch.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.trainable = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.cnot_chain = [(i, i + 1) for i in range(n_wires - 1)]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=x.shape[0],
                                device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.trainable):
            gate(qdev, wires=wire)
        for src, tgt in self.cnot_chain:
            tqf.cnot(qdev, wires=[src, tgt])
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# 3. Unified hybrid layer
# --------------------------------------------------------------------------- #

class UnifiedHybridLayer(nn.Module):
    """Hybrid dense‑to‑quantum layer.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input vector.
    hidden_dim : int
        Size of the hidden representation that feeds the quantum gates.
        Default is 2 to match the fraud layer input dimension.
    n_qubits : int
        Number of qubits used by each gate module.
    fraud_params : FraudLayerParameters | None
        Parameters for the photonic fraud transformation.  If *None* a
        neutral default is used.
    """
    def __init__(self,
                 n_features: int = 1,
                 hidden_dim: int = 2,
                 n_qubits: int = 2,
                 fraud_params: FraudLayerParameters | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, hidden_dim)

        # LSTM‑style quantum gates
        self.forget_gate = QuantumGate(n_qubits)
        self.input_gate = QuantumGate(n_qubits)
        self.update_gate = QuantumGate(n_qubits)
        self.output_gate = QuantumGate(n_qubits)

        # Linear maps that bring the hidden vector into the qubit space
        self.linear_forget = nn.Linear(hidden_dim, n_qubits)
        self.linear_input = nn.Linear(hidden_dim, n_qubits)
        self.linear_update = nn.Linear(hidden_dim, n_qubits)
        self.linear_output = nn.Linear(hidden_dim, n_qubits)

        # Fraud‑style post‑processing
        if fraud_params is None:
            fraud_params = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        self.fraud_layer = _fraud_layer_from_params(fraud_params, clip=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear feature extraction
        h = self.linear(x)

        # Quantum gate outputs
        f = torch.sigmoid(self.forget_gate(self.linear_forget(h)))
        i = torch.sigmoid(self.input_gate(self.linear_input(h)))
        g = torch.tanh(self.update_gate(self.linear_update(h)))
        o = torch.sigmoid(self.output_gate(self.linear_output(h)))

        # LSTM‑style cell state (no carry‑over in this simplified version)
        c = f * g
        h_new = o * torch.tanh(c)

        # Photonic fraud transformation
        out = self.fraud_layer(h_new)

        return out


def FCL() -> UnifiedHybridLayer:
    """Return the hybrid layer class that mirrors the original FCL API."""
    return UnifiedHybridLayer


__all__ = ["FCL", "UnifiedHybridLayer", "FraudLayerParameters"]
