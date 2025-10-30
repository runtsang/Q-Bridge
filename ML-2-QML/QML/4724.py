from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class FraudQuantumModel(tq.QuantumModule):
    """Quantum variational model that complements the photonic and CNN branches."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=40, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT(wires=[0, 1])

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            self.cnot(qdev)

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        # Encoder that maps a classical feature vector to a quantum state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{input_dim}xRy"]
        )
        self.q_layer = self.QLayer(input_dim)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.input_dim, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["FraudQuantumModel"]
