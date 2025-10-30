from __future__ import annotations

import pennylane as qml
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class FraudLayerParameters:
    """Parameters that mirror the classical layer but are used to initialise
    a variational photonic‑style circuit on a qubit device."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _flatten_params(p: FraudLayerParameters) -> List[float]:
    return [
        p.bs_theta,
        p.bs_phi,
        p.phases[0],
        p.phases[1],
        p.squeeze_r[0],
        p.squeeze_r[1],
        p.squeeze_phi[0],
        p.squeeze_phi[1],
        p.displacement_r[0],
        p.displacement_r[1],
        p.displacement_phi[0],
        p.displacement_phi[1],
        p.kerr[0],
        p.kerr[1],
    ]


class FraudDetection:
    """
    Quantum fraud‑detection model implemented with Pennylane.  The circuit emulates
    the photonic motif from the classical seed using a set of parametrised rotations
    and entangling gates.  All parameters are trainable via torch.autograd and
    optimised with a standard Adam optimiser.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device_name: str = "default.qubit",
        wires: int = 2,
    ) -> None:
        self.device = qml.device(device_name, wires=wires)
        self.params = nn.ParameterList()
        self.params.append(
            nn.Parameter(torch.tensor(_flatten_params(input_params), dtype=torch.float32))
        )
        for layer in layers:
            self.params.append(
                nn.Parameter(torch.tensor(_flatten_params(layer), dtype=torch.float32))
            )
        self.optimizer = torch.optim.Adam(self.params, lr=0.01)

        # Construct the qnode
        def circuit(x: Tensor, params: Tensor) -> Tensor:
            (
                bs_theta,
                bs_phi,
                ph0,
                ph1,
                sr0,
                sr1,
                sp0,
                sp1,
                dr0,
                dr1,
                dp0,
                dp1,
                k0,
                k1,
            ) = params

            # Encode input as rotations
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)

            # Photonic‑style gates realised with qubit primitives
            qml.Rot(bs_theta, bs_phi, 0.0, wires=0)
            qml.Rot(bs_phi, bs_theta, 0.0, wires=1)
            qml.RZ(ph0, wires=0)
            qml.RZ(ph1, wires=1)
            qml.RX(sr0, wires=0)
            qml.RX(sr1, wires=1)
            qml.RZ(dr0, wires=0)
            qml.RZ(dr1, wires=1)
            qml.RZZ(k0, wires=[0, 1])
            qml.RZZ(k1, wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        self.qnode = qml.QNode(circuit, self.device, interface="torch")

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluate the circuit for a single 2‑dimensional input.

        Parameters
        ----------
        x : Tensor
            Shape (2,) or (batch, 2).  The function is vectorised via
            the qnode's batch support.

        Returns
        -------
        Tensor
            Output of shape (batch, 1) after applying a sigmoid.
        """
        all_params = torch.cat(self.params)
        out = self.qnode(x, all_params)
        return torch.sigmoid(out)

    def kl_divergence(self) -> Tensor:
        """
        Simple KL between each parameter and a standard normal prior.
        """
        kl = 0.0
        for p in self.params:
            kl += 0.5 * torch.sum(p**2 - torch.log(p**2 + 1e-8) - 1.0)
        return kl

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        kl_weight: float = 1e-3,
    ) -> float:
        """
        Perform one optimisation step on a single example or batch.

        Parameters
        ----------
        x : Tensor
            Input features of shape (batch, 2).
        y : Tensor
            Binary labels of shape (batch, 1).
        kl_weight : float
            Weight of the KL divergence term.

        Returns
        -------
        loss.item() : float
            Scalar loss value after the step.
        """
        self.optimizer.zero_grad()
        logits = self.forward(x)
        loss = F.binary_cross_entropy(logits, y) + kl_weight * self.kl_divergence()
        loss.backward()
        self.optimizer.step()
        return loss.item()


__all__ = ["FraudLayerParameters", "FraudDetection"]
