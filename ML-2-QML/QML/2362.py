"""Quantum hybrid sampler‑regression module.

This module implements the same logical flow as the classical counterpart
but replaces the sampler with a parameterised quantum circuit and the
regression head with a variational quantum circuit.  The design follows
the structure of the reference `SamplerQNN` and `QuantumRegression` modules
while adding a second variational layer for regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq


class HybridSamplerRegression(tq.QuantumModule):
    """
    Quantum sampler‑regression network.

    Parameters
    ----------
    num_features : int
        Dimensionality of the classical input vector that will be encoded
        into the quantum state.
    num_wires : int
        Number of qubits used for the regression head.
    """

    class SamplerLayer(tq.QuantumModule):
        """Parameterised sampler that maps classical features to a state."""

        def __init__(self, num_features: int, num_wires: int):
            super().__init__()
            self.num_features = num_features
            self.num_wires = num_wires
            # Encode each feature into a Ry rotation on a dedicated qubit
            self.ry = tq.RY(has_params=True, trainable=True)
            # Entangle all qubits to allow correlations
            self.cx = tq.CNOT()

        def forward(self, qdev: tq.QuantumDevice):
            # Assume the input state is already prepared by the encoder
            # Apply feature‑dependent rotations
            for wire in range(self.num_wires):
                self.ry(qdev, wires=wire)
            # Add entanglement
            for i in range(self.num_wires - 1):
                self.cx(qdev, wires=(i, i + 1))

    class RegressionLayer(tq.QuantumModule):
        """Variational regression layer."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires

        # Encoder maps classical vector to a product state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_features}xRy"])
        self.sampler_layer = self.SamplerLayer(num_features, num_wires)
        self.regression_layer = self.RegressionLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Classical input of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data
        self.encoder(qdev, state_batch)
        # Sample‑style variational layer
        self.sampler_layer(qdev)
        # Regression variational layer
        self.regression_layer(qdev)
        # Measurement
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridSamplerRegression"]
