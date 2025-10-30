"""Hybrid fraud‑detection model: classical dense blocks followed by a photonic variational circuit.

The new architecture keeps the original two‑mode structure but adds
- a tunable classical dense layer that learns a linear embedding before the first quantum layer
- a small variational quantum circuit that replaces the **all‑pass** photonic layers
  (BS, S, D, K).  This allows the model to be trained with a quantum‑classical
  hybrid optimiser.
- ``penalty`` parameter that can be used to regularise squeezing and displacement
  amplitudes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import pennylane as qml

# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used for both ML and QML)."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
#  Quantum helper
# --------------------------------------------------------------------------- #
def _quantum_layer(params: FraudLayerParameters, clip: bool = False) -> qml.QNode:
    """Return a Pennylane QNode that implements a single photonic layer."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # Encode classical inputs into two qubits
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)

        # Beam splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Squeezing (approximated by RZ+RX for the default.qubit simulator)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_eff = r if not clip else max(min(r, 5.0), -5.0)
            qml.RX(r_eff, wires=i)
            qml.RZ(phi, wires=i)

        # Displacement (approximated by RX)
        for i, (d, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            d_eff = d if not clip else max(min(d, 5.0), -5.0)
            qml.RX(d_eff, wires=i)
            qml.RZ(phi, wires=i)

        # Kerr (approximated by RZ)
        for i, k in enumerate(params.kerr):
            k_eff = k if not clip else max(min(k, 1.0), -1.0)
            qml.RZ(k_eff, wires=i)

        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit

# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    A hybrid classical–quantum fraud‑detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first photonic layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the subsequent photonic layers.
    penalty : float, optional
        Weight of the regularisation term that penalises large squeezing
        values.  Default is 0.0 (no penalty).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.penalty = penalty

        # Classical dense embedding (matches the seed's linear layer)
        self.embed = nn.Linear(2, 2)

        # Quantum layers
        self.quantum_layers = nn.ModuleList(
            [_QuantumLayer(params, clip=True) for params in layers]
        )
        self.first_quantum = _QuantumLayer(input_params, clip=False)

        # Final output layer
        self.out = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical embedding
        x = torch.tanh(self.embed(x))

        # First quantum layer (no clipping)
        x = self.first_quantum(x)

        # Subsequent quantum layers (clipped)
        for qlayer in self.quantum_layers:
            x = qlayer(x)

        # Final linear mapping
        return self.out(x)

    def regularisation(self) -> torch.Tensor:
        """Return the penalty term based on squeezing amplitudes."""
        if self.penalty == 0.0:
            return torch.tensor(0.0, device=self.first_quantum.weight.device)
        penalty_term = 0.0
        for qlayer in self.quantum_layers:
            penalty_term += torch.sum(qlayer.squeezing_amplitude**2)
        return self.penalty * penalty_term

# --------------------------------------------------------------------------- #
#  Quantum layer as nn.Module
# --------------------------------------------------------------------------- #
class _QuantumLayer(nn.Module):
    """Wrap a Pennylane QNode inside a PyTorch Module."""

    def __init__(self, params: FraudLayerParameters, clip: bool) -> None:
        super().__init__()
        self.params = params
        self.clip = clip
        self.qnode = _quantum_layer(params, clip=clip)
        # Store squeezing amplitudes for regularisation
        self.squeezing_amplitude = torch.tensor(
            list(params.squeeze_r) + list(params.squeeze_phi),
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x)

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybrid",
]
