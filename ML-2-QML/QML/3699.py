"""Quantum backend for the hybrid fraud detection model.

The class :class:`FraudDetectionHybrid` builds two distinct quantum circuits from the
parameter tensor produced by the classical extractor:
1. A Strawberry Fields photonic program that mimics the layered photonic model.
2. A simple Qiskit circuit whose expectation value of the Pauli‑Y operator
   serves as an additional quantum feature.

Both circuits return a scalar expectation value; the two are stacked into a
2‑dimensional feature vector that is consumed by the classical regression head.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Tuple, List

# --- Parameter description ---------------------------------------------------------
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

# --- Photonic utilities ----------------------------------------------------------
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate, Fock

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: List[sf.ops.Modes], params: FraudLayerParameters, *, clip: bool) -> None:
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

def _build_sf_program(input_params: FraudLayerParameters,
                      layers: List[FraudLayerParameters]) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

# --- Qiskit utilities -------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
from qiskit.primitives import Estimator as QiskitEstimator

# --- Hybrid quantum module ---------------------------------------------------------
class FraudDetectionHybrid:
    """Quantum backend that returns a 2‑dimensional feature vector."""
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.estimator = QiskitEstimator()
        self.sf_engine = sf.Engine("fock", backend_options={"cutoff_dim": 4})

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        params : torch.Tensor
            shape (batch, 8*(num_layers+1)), where each slice encodes one
            photonic layer.  The first two entries are used for the Qiskit
            circuit; the rest are discarded for simplicity.
        Returns
        -------
        torch.Tensor
            shape (batch, 2) – photonic expectation + qubit expectation.
        """
        batch = params.shape[0]
        photonic_feats = torch.zeros(batch, dtype=torch.float32)
        qiskit_feats = torch.zeros(batch, dtype=torch.float32)

        for i in range(batch):
            # --- Photonic circuit ----------------------------------------------------
            layer_params: List[FraudLayerParameters] = []
            for l in range(self.num_layers + 1):
                start = l * 8
                end = start + 8
                raw = params[i, start:end]
                layer = FraudLayerParameters(
                    bs_theta=raw[0].item(),
                    bs_phi=raw[1].item(),
                    phases=(raw[2].item(), raw[3].item()),
                    squeeze_r=(raw[4].item(), raw[5].item()),
                    squeeze_phi=(raw[6].item(), raw[7].item()),
                    displacement_r=(0.0, 0.0),   # placeholder
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                )
                layer_params.append(layer)
            sf_prog = _build_sf_program(layer_params[0], layer_params[1:])
            result = self.sf_engine.run(sf_prog)
            # Expectation of photon number in mode 0
            sf_expect = result.state.expectation_value(Fock(0))
            photonic_feats[i] = torch.tensor(sf_expect, dtype=torch.float32)

            # --- Qiskit circuit ------------------------------------------------------
            theta = params[i, 0].item()
            phi = params[i, 1].item()
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            qc.rx(phi, 0)
            observable = Pauli('Y')
            est_res = self.estimator.run([qc], [observable]).result()
            qiskit_expect = est_res[0].values[0]
            qiskit_feats[i] = torch.tensor(qiskit_expect, dtype=torch.float32)

        return torch.stack([photonic_feats, qiskit_feats], dim=1)

__all__ = ["FraudDetectionHybrid"]
