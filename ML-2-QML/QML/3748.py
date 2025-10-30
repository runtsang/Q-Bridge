from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class FraudLayerParameters:
    """
    Parameters that describe a single photonic‑inspired layer.
    Reused in the quantum implementation to keep the API identical.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudDetectionHybrid(tq.QuantumModule):
    """
    Quantum‑enhanced fraud‑detection model that mimics a photonic circuit.
    Classical fraud features are encoded into qubits, a sequence of
    trainable quantum gates (approximating beam splitters, squeezers, etc.)
    is applied, and the measurement outcomes are linearly combined to
    produce a fraud score.
    """

    class PhotonicLayer(tq.QuantumModule):
        """
        A single photonic‑style layer implemented with conventional
        quantum gates.  Parameters are used to initialise the rotation
        angles of RX, RY, and RZ gates, providing an analog of
        beam‑splitters and squeezers.
        """
        def __init__(self, params: FraudLayerParameters, clip: bool = False, wires: Sequence[int] | None = None):
            super().__init__()
            self.params = params
            self.clip = clip
            self.wires = wires or list(range(2))
            # Trainable rotation gates
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Beam‑splitter analog: two RX rotations with angles derived from bs_theta/phi
            for wire in self.wires:
                theta = _clip(self.params.bs_theta, 5.0) if self.clip else self.params.bs_theta
                phi = _clip(self.params.bs_phi, 5.0) if self.clip else self.params.bs_phi
                self.rx(qdev, wires=wire, theta=theta)
                self.ry(qdev, wires=wire, theta=phi)
            # Phase shifters
            for i, phase in enumerate(self.params.phases):
                self.rz(qdev, wires=self.wires[i], theta=phase)
            # Squeezing and displacement are approximated by additional RZ rotations
            for i, (r, phi) in enumerate(zip(self.params.squeeze_r, self.params.squeeze_phi)):
                self.rz(qdev, wires=self.wires[i], theta=_clip(r, 5.0) if self.clip else r)
            for i, (r, phi) in enumerate(zip(self.params.displacement_r, self.params.displacement_phi)):
                self.rz(qdev, wires=self.wires[i], theta=_clip(r, 5.0) if self.clip else r)
            # Kerr nonlinearity analog (small rotation)
            for i, k in enumerate(self.params.kerr):
                self.rz(qdev, wires=self.wires[i], theta=_clip(k, 1.0) if self.clip else k)

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Sequence[FraudLayerParameters],
        num_wires: int = 2,
    ) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.layers = nn.ModuleList(
            [self.PhotonicLayer(input_params, clip=False, wires=list(range(num_wires)))]
            + [self.PhotonicLayer(p, clip=True, wires=list(range(num_wires))) for p in hidden_params]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum model.

        Parameters
        ----------
        state_batch : torch.Tensor
            Input features of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Fraud score of shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    @classmethod
    def from_random(
        cls,
        num_hidden: int,
        seed: int | None = None,
        num_wires: int = 2,
    ) -> "FraudDetectionHybrid":
        """
        Factory that creates a quantum model with randomly initialized photonic parameters.
        """
        rng = np.random.default_rng(seed)

        def rand_param() -> FraudLayerParameters:
            return FraudLayerParameters(
                bs_theta=rng.standard_normal(),
                bs_phi=rng.standard_normal(),
                phases=tuple(rng.standard_normal(2)),
                squeeze_r=tuple(rng.standard_normal(2)),
                squeeze_phi=tuple(rng.standard_normal(2)),
                displacement_r=tuple(rng.standard_normal(2)),
                displacement_phi=tuple(rng.standard_normal(2)),
                kerr=tuple(rng.standard_normal(2)),
            )

        input_params = rand_param()
        hidden_params = [rand_param() for _ in range(num_hidden)]
        return cls(input_params, hidden_params, num_wires=num_wires)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
