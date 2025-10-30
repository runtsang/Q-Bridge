"""Quantum hybrid self‑attention using TorchQuantum."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumHybridSelfAttention(tq.QuantumModule):
    """Self‑attention block implemented as a variational quantum circuit.

    The circuit encodes the classical image features into a quantum
    state, applies a trainable random layer followed by parametrised
    single‑qubit rotations and a controlled‑X gate.  The measurement
    outcomes are interpreted as attention weights.
    """

    class QLayer(tq.QuantumModule):
        """Variational layer that consumes external rotation and entangle
        parameters supplied by the caller.
        """

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            # Single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Two‑qubit entanglement
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(
            self,
            qdev: tq.QuantumDevice,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
        ) -> None:
            """Apply a random layer and parametrised gates."""
            self.random_layer(qdev)
            # Apply rotations per wire
            for w in range(self.n_wires):
                self.rx(qdev, wires=w, params=rotation_params[3 * w : 3 * w + 1])
                self.ry(qdev, wires=w, params=rotation_params[3 * w + 1 : 3 * w + 2])
                self.rz(qdev, wires=w, params=rotation_params[3 * w + 2 : 3 * w + 3])
            # Entangle adjacent wires
            for w in range(self.n_wires - 1):
                self.crx(
                    qdev,
                    wires=[w, w + 1],
                    params=entangle_params[w : w + 1],
                )

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps a 4×4 image patch into a 4‑qubit state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Forward pass of the quantum self‑attention block.

        Parameters
        ----------
        inputs
            Input image tensor of shape (B, C, H, W).
        rotation_params
            Rotational parameters for the variational layer,
            shape (n_wires * 3,).
        entangle_params
            Entanglement parameters for the variational layer,
            shape (n_wires - 1,).
        """
        bsz = inputs.shape[0]
        # Pool to a 4×4 patch to match the encoder size
        pooled = torch.nn.functional.avg_pool2d(inputs, 6).view(bsz, -1)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=inputs.device,
            record_op=True,
        )
        # Encode classical features
        self.encoder(qdev, pooled)
        # Apply variational layer with supplied parameters
        self.q_layer(qdev, rotation_params, entangle_params)
        # Measurement gives a vector of expectation values
        out = self.measure(qdev)
        return self.norm(out)


def SelfAttention() -> QuantumHybridSelfAttention:
    """Factory returning a quantum hybrid self‑attention module."""
    return QuantumHybridSelfAttention(n_wires=4)


__all__ = ["SelfAttention"]
