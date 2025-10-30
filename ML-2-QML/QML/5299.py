"""
Quantum modules that provide the variational circuits used by the hybrid
models in the ML module.

All quantum layers are implemented with torchquantum and expose a
`forward` method that accepts either a `torch.Tensor` (for state‑vector
encoders) or a `QuantumDevice`.  The modules are intentionally
lightweight so that they can be dropped into the classical code path
via lazy imports.
"""

from __future__ import annotations

import torch
import torchquantum as tq
import torchquantum.functional as tqf
from torch import nn
from typing import Tuple

# --------------------------------------------------------------------------- #
# 1. Quantum gate layer for LSTM gates
# --------------------------------------------------------------------------- #

class QGateLayer(tq.QuantumModule):
    """
    Small variational circuit that maps a classical vector to a
    quantum state and measures all qubits in the Pauli‑Z basis.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode classical features into rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Parameterised rotations
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        # Entangling chain
        self.cnot_chain = [
            tq.CNOT if i < n_wires - 1 else tq.CNOT
            for i in range(n_wires - 1)
        ]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of classical inputs to qubit measurements."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entangle wires in a linear chain
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
# 2. Quantum regression head
# --------------------------------------------------------------------------- #

class QRegressionLayer(tq.QuantumModule):
    """
    Variational circuit that produces a scalar regression output from a
    quantum state.  The circuit consists of a random layer followed by
    trainable RX/RY rotations and a final measurement.
    """
    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.num_wires = num_wires
        # Random feature map
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# 3. Quantum NAT (QFC) feature extractor
# --------------------------------------------------------------------------- #

class QNATLayer(tq.QuantumModule):
    """
    Quantum circuit that transforms a pooled image feature vector into
    a 4‑dimensional representation, mimicking the Quantum‑NAT paper.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        bsz = pooled.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=pooled.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])
        features = self.measure(qdev)
        return self.norm(features)

# --------------------------------------------------------------------------- #
# 4. Exposed factory functions
# --------------------------------------------------------------------------- #

def get_quantum_lstm_gate(n_wires: int) -> QGateLayer:
    """Return a quantum gate layer suitable for use in HybridQLSTM."""
    return QGateLayer(n_wires)

def get_quantum_regression_head(num_wires: int) -> QRegressionLayer:
    """Return a quantum regression head."""
    return QRegressionLayer(num_wires)

def get_quantum_nat_head() -> QNATLayer:
    """Return a quantum NAT feature extractor."""
    return QNATLayer()

__all__ = [
    "QGateLayer",
    "QRegressionLayer",
    "QNATLayer",
    "get_quantum_lstm_gate",
    "get_quantum_regression_head",
    "get_quantum_nat_head",
]
