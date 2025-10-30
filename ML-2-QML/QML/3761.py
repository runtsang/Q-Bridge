"""Quantum hybrid sampler network.

This implementation mirrors the classical counterpart but replaces the
fully‑connected head with a 4‑qubit variational circuit.  The circuit is
inspired by the SamplerQNN parameterized circuit and the QFCModel::QLayer.
It accepts two input parameters (typically a reduced representation of
the CNN feature map) and four trainable weight parameters.  The output is
a probability distribution over four measurement outcomes, ready to be
used as a quantum sampler.
"""

from __future__ import annotations

import pennylane as qml
import torch
from typing import Tuple


class HybridSamplerQNN:
    """Quantum variational sampler.

    Parameters
    ----------
    wires : int, default 4
        Number of qubits in the device.
    device : str, default "default.qubit"
        Backend device name.
    """

    def __init__(self, wires: int = 4, device: str = "default.qubit") -> None:
        self.wires = wires
        self.dev = qml.device(device, wires=wires)
        self._setup_qnode()

    def _setup_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode the two input parameters
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entanglement layer reminiscent of QFCModel::QLayer
            qml.CNOT(wires=[0, 1])
            # Variational block
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.RZ(weights[2], wires=2)
            qml.CRX(weights[3], wires=[0, 2])
            # Measurement: probabilities over all computational basis states
            return qml.probs(wires=range(self.wires))

        self.circuit = circuit

    def forward(self, inputs: Tuple[float, float], weights: Tuple[float, float, float, float]) -> torch.Tensor:
        """Run the circuit and return the probability distribution.

        Parameters
        ----------
        inputs : Tuple[float, float]
            Two classical parameters (e.g., aggregated CNN features).
        weights : Tuple[float, float, float, float]
            Four trainable variational parameters.
        """
        return self.circuit(torch.tensor(inputs, dtype=torch.float32),
                            torch.tensor(weights, dtype=torch.float32))


__all__ = ["HybridSamplerQNN"]
