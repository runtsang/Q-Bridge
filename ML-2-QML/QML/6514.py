"""Photonic‑quantum fraud‑detection model with a classical classifier head.

The class builds a Strawberry‑Fields program that mirrors the photonic layer
definition of :class:`FraudDetectionParams`.  After simulation the program
returns expectation values of the photon‑number operator for each mode.
These quantum features are subsequently fed into a lightweight PyTorch
classifier that produces the final fraud probability.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

@dataclasses.dataclass
class FraudDetectionParams:
    """Parameters for one photonic block."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [−bound, bound]."""
    return max(-bound, min(bound, value))

def _apply_layer(modes: list, params: FraudDetectionParams, *, clip: bool) -> None:
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

def build_photonic_program(
    input_params: FraudDetectionParams,
    layers: Iterable[FraudDetectionParams],
) -> sf.Program:
    """Create a Strawberry‑Fields program with the given photonic layers."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

class FraudDetectionHybridQuantum(nn.Module):
    """Photonic circuit followed by a classical classifier.

    Parameters
    ----------
    input_params : FraudDetectionParams
        Parameters for the first photonic layer.
    layers : Iterable[FraudDetectionParams]
        Parameters for the remaining layers.
    clf_hidden : int
        Size of the hidden layer in the PyTorch classifier head.
    """

    def __init__(
        self,
        input_params: FraudDetectionParams,
        layers: Iterable[FraudDetectionParams],
        clf_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.program = build_photonic_program(input_params, layers)
        self.engine = Engine("fock", backend_options={"cutoff_dim": 8})
        # Classical classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2, clf_hidden),
            nn.ReLU(),
            nn.Linear(clf_hidden, 1),
        )

    def _quantum_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the photonic program for each row in ``inputs`` and return photon‑number expectations."""
        batch_size = inputs.shape[0]
        feats = np.zeros((batch_size, 2), dtype=np.float32)
        for i, row in enumerate(inputs.detach().cpu().numpy()):
            # Bind the first mode to the first feature, second mode to second feature
            self.program.input_vals = [row[0], row[1]]
            result = self.engine.run(self.program)
            state = result.state
            feats[i, 0] = state.expectation_value("n", [0])
            feats[i, 1] = state.expectation_value("n", [1])
        return torch.from_numpy(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probability after photonic feature extraction and classical head."""
        q_feats = self._quantum_features(x)
        return self.classifier(q_feats)

__all__ = ["FraudDetectionParams", "build_photonic_program", "FraudDetectionHybridQuantum"]
