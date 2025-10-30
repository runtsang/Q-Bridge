"""Quantum fraud detection model combining a Strawberry Fields photonic circuit
with a TorchQuantum quanvolutional filter.

The module defines a ``FraudDetector`` class that inherits from
``torchquantum.QuantumModule``.  It accepts a list of
``FraudLayerParameters`` to build a hybrid photonic–qubit circuit.
The class exposes a ``forward`` method that takes a batch of classical
feature vectors, encodes them, applies a quantum convolution, measures,
and produces a fraud score.

The photonic circuit is built with the same parameters as in the classical
seed, but the measurement results are fed into a linear head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torchquantum as tq
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# 1. Photonic‑inspired parameters (shared with the ML seed)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic‑inspired linear layer."""
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


def _apply_photonic_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
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
    """Create a Strawberry Fields program that mirrors the photonic circuit."""
    program = sf.Program(2)
    with program.context as q:
        _apply_photonic_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_photonic_layer(q, layer, clip=True)
    return program


# --------------------------------------------------------------------------- #
# 2. Quantum convolution (Quanvolution) filter
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, qdev.state)
        self.q_layer(qdev)
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# 3. Hybrid quantum fraud detector
# --------------------------------------------------------------------------- #
class FraudDetector(tq.QuantumModule):
    """Hybrid quantum fraud detector.

    Parameters
    ----------
    layer_params : Iterable[FraudLayerParameters]
        Parameters for the photonic‑inspired layers that will be embedded in the quantum circuit.
    """

    def __init__(self, layer_params: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.layer_params = list(layer_params)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4xRy"])
        self.q_filter = QuantumQuanvolutionFilter()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(4 * 14 * 14, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass that encodes the classical input into a quantum state,
        applies the quanvolution filter, measures, and maps to a fraud score.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of feature vectors of shape (batch, 4*14*14).

        Returns
        -------
        torch.Tensor
            Fraud score of shape (batch, 1).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=state_batch.device)

        # 1. Encode the input features into the qubit state
        self.encoder(qdev, state_batch)

        # 2. Apply the quantum convolution filter
        self.q_filter(qdev)

        # 3. Measure to obtain classical features
        features = self.measure(qdev)

        # 4. Pass through a classical linear head
        return self.head(features).squeeze(-1)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetector",
]
