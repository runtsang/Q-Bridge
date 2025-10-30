from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate
from dataclasses import dataclass
from typing import List, Tuple


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


class FraudDetectionHybrid:
    """
    Photonic circuit builder and Qiskit sampler circuit for fraud detection.
    Parameters are supplied by the classical encoder.
    """

    def __init__(self, num_layers: int = 3) -> None:
        self.num_layers = num_layers

    def build_program(self, params_list: List[FraudLayerParameters]) -> sf.Program:
        """Return a Strawberry Fields program implementing the photonic layers."""
        prog = sf.Program(2)
        with prog.context as q:
            # Input layer (no clipping)
            self._apply_layer(q, params_list[0], clip=False)
            # Hidden layers
            for params in params_list[1:]:
                self._apply_layer(q, params, clip=True)
        return prog

    def _apply_layer(self, modes, params: FraudLayerParameters, *, clip: bool) -> None:
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

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def build_sampler_circuit(self, input_params: Tuple[float, float],
                              weight_params: Tuple[float, float, float, float]) -> "qiskit.circuit.QuantumCircuit":
        """
        Construct a Qiskit quantum circuit for the sampler QNN.
        """
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        # Encode classical input
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)
        # Entanglement
        qc.cx(0, 1)
        # Parameterized rotations
        qc.ry(weight_params[0], 0)
        qc.ry(weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[2], 0)
        qc.ry(weight_params[3], 1)
        return qc


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
