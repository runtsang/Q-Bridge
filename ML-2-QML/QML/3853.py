"""
QuantumRegressionHybrid: Quantum refinement block for the hybrid regression model.
Author: Auto‑Generated
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

class QuantumRegressionHybrid(tq.QuantumModule):
    """
    Variational quantum refinement block that maps a classical feature vector
    to a scalar via expectation values.

    The block performs:
        * Encoding of the classical vector into a quantum state.
        * A parameterised variational circuit (random layer + RX/RY).
        * Measurement of Pauli‑Z on all qubits.
        * A linear readout that produces a single output value.
    """

    class QLayer(tq.QuantumModule):
        """
        Parameterised variational circuit:
            RandomLayer -> RX -> RY on each qubit.
        """
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

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires

        # Encode classical features into the quantum state
        # The chosen encoder applies an Ry rotation per qubit.
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )

        # Variational layer
        self.q_layer = self.QLayer(num_wires)

        # Measure Pauli‑Z on all wires
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Linear readout
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape (batch, num_wires) representing classical features.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch,) containing the predicted scalar.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        # Encode the classical vector into the quantum state
        self.encoder(qdev, state_batch)

        # Apply variational circuit
        self.q_layer(qdev)

        # Measure expectation values
        features = self.measure(qdev)

        # Readout
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionHybrid"]
