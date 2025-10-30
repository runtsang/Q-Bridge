"""Quantum‑centric fraud‑detection module that fuses a photonic circuit with a variational classifier.

The module implements a Strawberry‑Fields program that mirrors the photonic layer stack
and then appends a TorchQuantum variational ansatz to produce a classification score.
The variational circuit is trained end‑to‑end and can be executed on any compatible
back‑end (CPU simulator, QPU, etc.).  This design demonstrates how a classical photonic
simulation can be coupled with a quantum kernel for a hybrid inference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
#  Photonic layer definition (same as the ML side but in a quantum context)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry‑Fields program for the photonic fraud‑detection stack."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

# --------------------------------------------------------------------------- #
#  Variational classifier (TorchQuantum)
# --------------------------------------------------------------------------- #

class FraudVariationalAnsatz(tq.QuantumModule):
    """
    Parameterised ansatz that encodes a 2‑dimensional input and produces a
    single expectation value used as a classification score.
    """

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Simple entangling circuit: Ry → CX → Ry
        self.ansatz = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [], "func": "cx", "wires": [0, 2]},
            {"input_idx": [], "func": "cx", "wires": [1, 3]},
            {"input_idx": [], "func": "ry", "wires": [2]},
            {"input_idx": [], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def expectation(self, x: torch.Tensor) -> torch.Tensor:
        self.forward(self.q_device, x)
        # Measure PauliZ on the last wire
        return self.q_device.expectation([("Z", self.n_wires - 1)])

class QuantumFraudDetectionHybrid:
    """
    End‑to‑end quantum fraud‑detection model.

    The forward pass first runs the photonic program on a Strawberry‑Fields simulator,
    then feeds the resulting mode amplitudes into a TorchQuantum variational circuit.
    The expectation value is passed through a sigmoid to yield a fraud probability.
    """

    def __init__(
        self,
        photonic_params: FraudLayerParameters,
        photonic_layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.photonic_prog = build_fraud_detection_program(photonic_params, photonic_layers)
        self.ansatz = FraudVariationalAnsatz()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Fraud probability in (0, 1).
        """
        # Run the photonic circuit on a Strawberry‑Fields simulator
        eng = sf.Engine("gaussian", backend_options={"cutoff_dim": 10})
        prog = self.photonic_prog
        # Encode classical data into the first mode via displacement
        prog.context.displace(x[:, 0], 0)
        prog.context.displace(x[:, 1], 1)
        result = eng.run(prog)
        # Extract mode amplitudes (use first two modes as features)
        modes = result.state.mode_occupations()  # shape: (batch, 2)
        # Feed into the variational ansatz
        logits = self.ansatz.expectation(modes)
        return torch.sigmoid(logits)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudVariationalAnsatz",
    "QuantumFraudDetectionHybrid",
]
