from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QLayer(tq.QuantumModule):
    """
    Variational quantum layer built from a random circuit followed by
    trainable single‑qubit rotations and a few entangling gates.
    """
    def __init__(self, n_wires: int, n_ops: int = 30) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.hadamard = tq.Hadamard()
        self.sx = tq.SX()
        self.cnot = tq.CNOT()

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.crx(qdev, wires=[0, 3])
        self.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        self.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        self.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

class QuantumNATHybrid(tq.QuantumModule):
    """
    Quantum head that encodes the first ``n_qubits`` features into a quantum state,
    applies a variational layer, and measures a Pauli‑Z expectation.  The output
    is a vector of expectation values that can be post‑processed by a classical
    sigmoid layer to produce probabilities.
    """
    def __init__(self, n_qubits: int = 4, n_ops: int = 30) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Encoding ansatz inspired by the kernel method
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.n_qubits}x4_ryzxy"]
        )
        self.q_layer = QLayer(self.n_qubits, n_ops)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_qubits)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, d) – input features. Only the first ``n_qubits`` are used for encoding.
        Returns a tensor of shape (batch, n_qubits) containing Pauli‑Z expectations.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits, bsz=bsz, device=x.device, record_op=True
        )
        # Encode first n_qubits features with Ry gates
        encoded = x[:, :self.n_qubits]
        self.encoder(qdev, encoded)
        self.q_layer(qdev)
        out = self.measure(qdev)  # (batch, n_qubits)
        return self.norm(out)

__all__ = ["QuantumNATHybrid"]
