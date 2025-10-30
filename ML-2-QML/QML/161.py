"""Quantum photonic fraud detection circuit with variational ansatz and photon‑number measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import torch


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


def _flatten(params: FraudLayerParameters) -> torch.Tensor:
    return torch.tensor(
        [
            params.bs_theta,
            params.bs_phi,
            params.phases[0],
            params.phases[1],
            params.squeeze_r[0],
            params.squeeze_r[1],
            params.squeeze_phi[0],
            params.squeeze_phi[1],
            params.displacement_r[0],
            params.displacement_r[1],
            params.displacement_phi[0],
            params.displacement_phi[1],
            params.kerr[0],
            params.kerr[1],
        ],
        dtype=torch.float32,
    )


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Create a variational photonic circuit that returns a fraud score."""
    dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=10)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(param_vector: torch.Tensor) -> torch.Tensor:
        idx = 0

        def get_next(n: int) -> torch.Tensor:
            nonlocal idx
            vals = param_vector[idx : idx + n]
            idx += n
            return vals

        def _apply_layer(
            bs_theta,
            bs_phi,
            phase0,
            phase1,
            squeeze_r0,
            squeeze_r1,
            squeeze_phi0,
            squeeze_phi1,
            disp_r0,
            disp_r1,
            disp_phi0,
            disp_phi1,
            kerr0,
            kerr1,
            clip: bool,
        ) -> None:
            # Beam splitter
            qml.BSgate(bs_theta, bs_phi, wires=[0, 1])
            # Phase shifters
            qml.Rgate(phase0, wires=0)
            qml.Rgate(phase1, wires=1)
            # Squeezing
            if clip:
                squeeze_r0 = torch.clamp(squeeze_r0, -5.0, 5.0)
                squeeze_r1 = torch.clamp(squeeze_r1, -5.0, 5.0)
            qml.Sgate(squeeze_r0, squeeze_phi0, wires=0)
            qml.Sgate(squeeze_r1, squeeze_phi1, wires=1)
            # Beam splitter again
            qml.BSgate(bs_theta, bs_phi, wires=[0, 1])
            # Phase shifters again
            qml.Rgate(phase0, wires=0)
            qml.Rgate(phase1, wires=1)
            # Displacement
            if clip:
                disp_r0 = torch.clamp(disp_r0, -5.0, 5.0)
                disp_r1 = torch.clamp(disp_r1, -5.0, 5.0)
            qml.Dgate(disp_r0, disp_phi0, wires=0)
            qml.Dgate(disp_r1, disp_phi1, wires=1)
            # Kerr
            if clip:
                kerr0 = torch.clamp(kerr0, -1.0, 1.0)
                kerr1 = torch.clamp(kerr1, -1.0, 1.0)
            qml.KerrGate(kerr0, wires=0)
            qml.KerrGate(kerr1, wires=1)

        # First layer (unclipped)
        bs_theta, bs_phi = get_next(2)
        phase0, phase1 = get_next(2)
        squeeze_r0, squeeze_r1 = get_next(2)
        squeeze_phi0, squeeze_phi1 = get_next(2)
        disp_r0, disp_r1 = get_next(2)
        disp_phi0, disp_phi1 = get_next(2)
        kerr0, kerr1 = get_next(2)
        _apply_layer(
            bs_theta,
            bs_phi,
            phase0,
            phase1,
            squeeze_r0,
            squeeze_r1,
            squeeze_phi0,
            squeeze_phi1,
            disp_r0,
            disp_r1,
            disp_phi0,
            disp_phi1,
            kerr0,
            kerr1,
            clip=False,
        )

        # Subsequent layers (clipped)
        for _ in range(len(layers)):
            bs_theta, bs_phi = get_next(2)
            phase0, phase1 = get_next(2)
            squeeze_r0, squeeze_r1 = get_next(2)
            squeeze_phi0, squeeze_phi1 = get_next(2)
            disp_r0, disp_r1 = get_next(2)
            disp_phi0, disp_phi1 = get_next(2)
            kerr0, kerr1 = get_next(2)
            _apply_layer(
                bs_theta,
                bs_phi,
                phase0,
                phase1,
                squeeze_r0,
                squeeze_r1,
                squeeze_phi0,
                squeeze_phi1,
                disp_r0,
                disp_r1,
                disp_phi0,
                disp_phi1,
                kerr0,
                kerr1,
                clip=True,
            )

        # Photon‑number difference as fraud score proxy
        return qml.expval(qml.PauliZ(0)) - qml.expval(qml.PauliZ(1))

    # Flatten all parameters into a single vector
    flat_input = _flatten(input_params)
    flat_layers = [_flatten(l) for l in layers]
    flat_vector = torch.cat([flat_input] + flat_layers)

    # Return a callable that evaluates the circuit with the fixed parameters
    def qnode() -> torch.Tensor:
        return circuit(flat_vector)

    return qnode


class FraudDetection:
    """Shared interface for quantum fraud detection model."""
    @staticmethod
    def build_qnode(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> qml.QNode:
        return build_fraud_detection_program(input_params, layers)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetection"]
