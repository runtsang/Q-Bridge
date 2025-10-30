"""Hybrid quantum fraud detection model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import torch
from torch import nn

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

class FraudDetectionModel(nn.Module):
    """Quantum‑classical hybrid fraud detection model.

    A classical linear pre‑processor feeds a PennyLane variational circuit
    that mirrors the photonic layer parameters.  The variational parameters
    are trainable and initialized from the photonic values.  A final
    classical linear head produces the scalar output.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        n_qubits: int = 2,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.device = device

        # Classical pre‑processor
        self.pre = nn.Linear(2, 2)

        # Store layers for mapping parameters
        self.layers = [input_params] + list(layers)

        # Initialise variational parameters from photonic layers
        self._init_variational_params()

        # Variational circuit
        self.qnode = qml.QNode(self._quantum_circuit, qml.device(device, wires=n_qubits))

        # Classical post‑processor
        self.post = nn.Linear(3, 1)

    def _init_variational_params(self) -> None:
        """Flatten all photonic parameters into a trainable tensor."""
        params = []
        for layer in self.layers:
            params.extend(
                [
                    layer.bs_theta,
                    layer.bs_phi,
                    *layer.phases,
                    *layer.squeeze_r,
                    *layer.squeeze_phi,
                    *layer.displacement_r,
                    *layer.displacement_phi,
                    *layer.kerr,
                ]
            )
        self.var_params = nn.Parameter(torch.tensor(params, dtype=torch.float32))

    def _quantum_circuit(self, *var_params) -> float:
        """Variational circuit mirroring the photonic layer structure."""
        idx = 0
        for layer in self.layers:
            # Beam splitter (approximated by a two‑qubit rotation)
            theta = var_params[idx]
            phi = var_params[idx + 1]
            idx += 2
            qml.CNOT(wires=[0, 1])
            qml.RZ(phi, wires=0)
            qml.RZ(theta, wires=1)

            # Phase shift
            phase = var_params[idx]
            idx += 1
            qml.RZ(phase, wires=0)

            # Squeeze (approximated by RY and RZ)
            r = var_params[idx]
            phi_s = var_params[idx + 1]
            idx += 2
            qml.RY(r, wires=0)
            qml.RZ(phi_s, wires=0)

            # Displacement (approximated by RX)
            d = var_params[idx]
            phi_d = var_params[idx + 1]
            idx += 2
            qml.RX(d, wires=0)
            qml.RZ(phi_d, wires=0)

            # Kerr (non‑linear, approximated by RZ)
            k = var_params[idx]
            idx += 1
            qml.RZ(k, wires=0)

        # Measurement: expectation of PauliZ on first qubit
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: classical → quantum → classical."""
        x = self.pre(x)
        # Quantum output
        qout = self.qnode(*self.var_params)
        qout = torch.tensor(qout, dtype=x.dtype, device=x.device)
        # Concatenate classical and quantum features
        out = torch.cat([x, qout.unsqueeze(-1)], dim=-1)
        return self.post(out)
