"""Quantum kernel and photonic fraud detection circuit.

This module implements two complementary quantum components:

* ``FraudDetectionHybrid`` – a TorchQuantum module that evaluates a variational kernel between an input vector and a set of support vectors.
  The ansatz uses single‑qubit rotations parameterised by the input data and a fixed depth of 4 qubits.

* ``build_photonic_fraud_circuit`` – a Strawberry Fields program that mirrors the photonic fraud detection circuit from the classical seed.
  The circuit can be executed on the Strawberry Fields simulator or on a photonic hardware backend.

Both components are exposed via ``__all__`` for easy import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

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

def build_photonic_fraud_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# --------------------------------------------------------------------------- #
# Quantum kernel implementation
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(tq.QuantumModule):
    """
    TorchQuantum module that evaluates a variational kernel between an input
    vector and a fixed set of support vectors.

    The ansatz consists of a single‑qubit Ry rotation for each feature,
    followed by a reverse sequence with negative parameters to encode the
    second vector.  The kernel value is the overlap of the resulting state
    with the all‑zero computational basis state.
    """

    def __init__(self, support_vectors: torch.Tensor):
        super().__init__()
        self.support_vectors = support_vectors
        self.n_wires = support_vectors.shape[1]
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def _apply_ansatz(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for i, wire in enumerate(range(self.n_wires)):
            param = x[:, i]
            tq.RY(param) | wire
        for i, wire in reversed(list(enumerate(range(self.n_wires)))):
            param = -y[:, i]
            tq.RY(param) | wire

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel similarities between ``x`` and all support vectors.
        """
        sims = []
        for sv in self.support_vectors:
            self._apply_ansatz(self.q_device, x, sv.unsqueeze(0))
            sims.append(torch.abs(self.q_device.states.view(-1)[0]))
        return torch.stack(sims, dim=-1)

def build_quantum_kernel(
    support_vectors: torch.Tensor,
) -> FraudDetectionHybrid:
    """
    Helper that constructs a :class:`FraudDetectionHybrid` for reuse in
    hybrid models.
    """
    return FraudDetectionHybrid(support_vectors)

__all__ = [
    "FraudLayerParameters",
    "build_photonic_fraud_circuit",
    "FraudDetectionHybrid",
    "build_quantum_kernel",
]
