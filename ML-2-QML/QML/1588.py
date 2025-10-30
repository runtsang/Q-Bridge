"""Quantum embedding for fraud detection using a variational photonic circuit.

The module implements a PennyLane QNode that maps the classical parameters
to a feature vector of expectation values.  The circuit is simple enough to
run on a simulator but can be ported to real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pennylane as qml
import torch

__all__ = ["QuantumFraudParameters", "QuantumFraudEmbedding"]

@dataclass
class QuantumFraudParameters:
    """Parameters that control the variational photonic circuit."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class QuantumFraudEmbedding(torch.nn.Module):
    """A PyTorch module that returns a quantum‑derived embedding vector.

    The embedding is produced by a PennyLane QNode that runs a two‑mode
    circuit with a small number of parameterised gates.  The output
    consists of the expectation values of Pauli‑Z on each mode.
    """
    def __init__(self, params: QuantumFraudParameters, device_name: str = "default.qubit") -> None:
        super().__init__()
        self.params = params
        self.device = qml.device(device_name, wires=2)
        self._qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self) -> Tuple[float, float]:
        # Beam splitter
        qml.BSgate(self.params.bs_theta, self.params.bs_phi, wires=[0, 1])
        # Phase shifts
        for i, ph in enumerate(self.params.phases):
            qml.Rgate(ph, wires=i)
        # Squeezing
        for i, (r, phi) in enumerate(zip(self.params.squeeze_r, self.params.squeeze_phi)):
            qml.Sgate(r, phi, wires=i)
        # Displacement
        for i, (r, phi) in enumerate(zip(self.params.displacement_r, self.params.displacement_phi)):
            qml.Dgate(r, phi, wires=i)
        # Kerr
        for i, k in enumerate(self.params.kerr):
            qml.Kgate(k, wires=i)
        # Observables
        return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)))

    def forward(self, _: torch.Tensor = None) -> torch.Tensor:
        """Return a tensor of shape (2,) containing the expectation values."""
        return self._qnode()
