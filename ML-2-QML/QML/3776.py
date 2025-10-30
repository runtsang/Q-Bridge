"""Quantum feature extractor for the hybrid model.

Uses a 4‑qubit variational circuit with RX rotations driven by the input
features, a shallow entangling layer, and a Pauli‑Z measurement that
produces a 4‑dimensional feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumHybridNat(tq.QuantumModule):
    """Quantum module that encodes 4 input values into a 4‑qubit state and
    returns the measurement outcomes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder: each input element drives an RX rotation on a separate qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)
            ]
        )
        # Trainable rotation layers
        self.rotation = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
        )
        # Entangling layer: chain of CNOTs
        self.cnot_layers = nn.ModuleList(
            [tq.CNOT() for _ in range(self.n_wires - 1)]
        )
        # Measure all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice | None = None) -> torch.Tensor:
        """Return a 4‑dimensional measurement vector.

        Expected input shape: (batch, 4) or (4,).  If no quantum device is
        supplied, a simulator with the appropriate batch size is created.
        """
        if qdev is None:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.size(0), device=x.device)
        # Encode classical data into rotations
        self.encoder(qdev, x)
        # Apply trainable rotations
        for gate in self.rotation:
            gate(qdev)
        # Entangle qubits
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        # Return measurement outcomes
        return self.measure(qdev)
