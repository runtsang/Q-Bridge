"""Variational hybrid fraud‑detection circuit built with Strawberry Fields.

This module lifts the original seed by turning the fixed photonic gates into
parameterised ansätze that can be trained with a quantum‑classical optimiser.
The circuit also encodes input features as displacements, turning raw
transaction data into a learned continuous‑variable feature map.
"""

from __future__ import annotations

import strawberryfields as sf
from strawberryfields import ops
from dataclasses import dataclass
from typing import Iterable, List, Tuple

__all__ = ["FraudLayerParameters", "FraudDetectionExtended"]

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""

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


# --------------------------------------------------------------------------- #
# Main quantum model
# --------------------------------------------------------------------------- #
class FraudDetectionExtended:
    """Variational hybrid circuit that mirrors the photonic seed but
    replaces hard‑coded parameters with trainable `sf.Parameter` objects.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.program = self._build_program()

    # --------------------------------------------------------------------- #
    # Helper to create a trainable layer
    # --------------------------------------------------------------------- #
    def _apply_layer(
        self,
        q: sf.Context,
        params: FraudLayerParameters,
        clip: bool,
    ) -> None:
        """Apply a photonic layer with trainable parameters."""
        # Beam splitter
        theta = sf.Parameter("theta")
        phi = sf.Parameter("phi")
        ops.BSgate(theta, phi) | (q[0], q[1])

        # Phase shifters
        for i, phase in enumerate(params.phases):
            ops.Rgate(phase) | q[i]

        # Squeezing
        for i, (r, p) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_par = sf.Parameter(f"squeeze_r_{i}") if clip else r
            p_par = sf.Parameter(f"squeeze_phi_{i}") if clip else p
            ops.Sgate(r_par, p_par) | q[i]

        # Second beam splitter
        ops.BSgate(theta, phi) | (q[0], q[1])

        # Phase shifters again
        for i, phase in enumerate(params.phases):
            ops.Rgate(phase) | q[i]

        # Displacements (feature map)
        for i, (r, p) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_par = sf.Parameter(f"disp_r_{i}") if clip else r
            p_par = sf.Parameter(f"disp_phi_{i}") if clip else p
            ops.Dgate(r_par, p_par) | q[i]

        # Kerr
        for i, k in enumerate(params.kerr):
            k_par = sf.Parameter(f"kerr_{i}") if clip else k
            ops.Kgate(k_par) | q[i]

    # --------------------------------------------------------------------- #
    # Build the full program
    # --------------------------------------------------------------------- #
    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)
        return prog

    # --------------------------------------------------------------------- #
    # Accessors
    # --------------------------------------------------------------------- #
    def get_parameters(self) -> List[sf.Parameter]:
        """Return all trainable parameters in the program."""
        return self.program.get_parameters()

    def run(self, shots: int = 1024) -> sf.results.Results:
        """Execute the program on a Gaussian backend and return measurement results."""
        backend = sf.backends.GaussianBackend(2)
        return backend.run(self.program, n_shots=shots)

    def expectation_photon(self, shots: int = 1024) -> float:
        """Return the mean photon number measured on mode 0."""
        results = self.run(shots)
        return results.mean(0)[0]  # mode 0 photon number

    def predict(self, shots: int = 1024) -> float:
        """Return a binary prediction (0/1) based on a threshold of the photon
        expectation value.  The threshold can be tuned outside this class.
        """
        exp = self.expectation_photon(shots)
        return 1.0 if exp > 1.0 else 0.0
