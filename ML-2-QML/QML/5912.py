"""
FraudDetectionHybrid quantum component.

Provides a QuantumFraudCircuit class that builds a Strawberry‑Fields program
matching the FraudLayerParameters.  The execute() method accepts a
batch of 2‑dim inputs, injects them as displacements into the first mode,
runs the program and returns the photon‑number expectation values of
both modes as a torch tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, NumberOp

@dataclass
class FraudLayerParameters:
    """
    Parameters describing a single layer of the fraud model.
    """
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class QuantumFraudCircuit:
    """
    Quantum circuit that mirrors the classical FraudLayerParameters.
    The `execute` method takes a batch of 2‑dim inputs, injects them
    as displacements into the first mode, runs the program and returns
    the photon‑number expectation values of both modes.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        backend: str = "fock",
        cutoff_dim: int = 10,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.engine = Engine(backend, backend_options={"cutoff_dim": cutoff_dim})

    def _build_program(self, disp: Tuple[float, float]) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            # Input layer
            self._apply_layer(q, self.input_params, clip=False, disp=disp)
            # Hidden layers
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True, disp=disp)
        return prog

    def _apply_layer(
        self,
        modes: Sequence,
        params: FraudLayerParameters,
        *,
        clip: bool,
        disp: Tuple[float, float],
    ) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]
        # Inject the input displacement into the first mode
        Dgate(disp[0], disp[1]) | modes[0]

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def execute(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Execute the circuit for a batch of 2‑dim input vectors.
        Each input is treated as a displacement (r, phi) for the first mode.
        Returns a tensor of shape (batch, 2) containing the mean photon number
        of each mode after the circuit.
        """
        batch = inputs.shape[0]
        results = []
        for idx in range(batch):
            r, phi = inputs[idx].tolist()
            prog = self._build_program((r, phi))
            result = self.engine.run(prog)
            mean_photon = [
                result.expectation_value(NumberOp, mode) for mode in range(2)
            ]
            results.append(mean_photon)
        return torch.tensor(results, dtype=torch.float32)

__all__ = ["FraudLayerParameters", "QuantumFraudCircuit"]
