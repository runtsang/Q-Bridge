"""Hybrid variational fraud detection model using Pennylane Gaussian device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic‑inspired variational layer."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel(nn.Module):
    """
    A hybrid model that encodes classical inputs into a Gaussian
    circuit, measures photon‑number expectations, and applies a
    classical linear head.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first variational layer.
    layers : Iterable[FraudLayerParameters]
        Additional layers that share the same gate set.
    device : str, optional
        Pennylane device name (default: ``"default.gaussian"``).
    shots : int, optional
        Number of shots for the measurement (default: 1024).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "default.gaussian",
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.shots = shots
        self.params = [input_params] + list(layers)

        # Classical read‑out head
        self.weight = nn.Parameter(torch.randn(2, 1))
        self.bias = nn.Parameter(torch.randn(1))

        self._build_qnode()

    def _build_qnode(self) -> None:
        dev = qml.device(self.device_name, wires=2, shots=self.shots)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode two real inputs as displacements on the two modes
            qml.Dgate(inputs[0], 0.0, wires=0)
            qml.Dgate(inputs[1], 0.0, wires=1)

            # Sequentially apply photonic‑inspired gates
            for layer in self.params:
                qml.BSgate(layer.bs_theta, layer.bs_phi, wires=[0, 1])
                for i, phase in enumerate(layer.phases):
                    qml.Rgate(phase, wires=i)
                for i, (r, phi) in enumerate(zip(layer.squeeze_r, layer.squeeze_phi)):
                    qml.Sgate(r, phi, wires=i)
                for i, (r, phi) in enumerate(zip(layer.displacement_r, layer.displacement_phi)):
                    qml.Dgate(r, phi, wires=i)
                for i, k in enumerate(layer.kerr):
                    qml.Kgate(k, wires=i)

            # Measure photon‑number expectation on each mode
            return [qml.expval(qml.NumberOperator(wires=i)) for i in range(2)]

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch, 2)`` containing the two classical features.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` with the fraud‑risk score.
        """
        batch_size = inputs.shape[0]
        outputs = []
        for x in inputs:
            meas = self.circuit(x)
            meas = torch.stack(meas)
            out = self.weight.t() @ meas.unsqueeze(1) + self.bias
            outputs.append(out.squeeze())
        return torch.stack(outputs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(layers={len(self.params)}, "
            f"shots={self.shots}, device={self.device_name})"
        )


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
