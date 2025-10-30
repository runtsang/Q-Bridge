import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumSelfAttention(tq.QuantumModule):
    """
    Quantum self‑attention sub‑circuit using RX/RZ rotations and CRX entanglement.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.rx = tq.RX(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Apply a layer of single‑qubit rotations
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.rz(qdev, wires=w)
        # Entangle adjacent qubits with a CRX gate
        for w in range(self.n_wires - 1):
            self.crx(qdev, wires=[w, w + 1])

class QuantumNATGen221(tq.QuantumModule):
    """
    Quantum counterpart of the hybrid Quantum‑NAT model.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps classical features to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        # Variational block with a random layer and a quantum self‑attention sub‑circuit
        self.q_layer = self._build_q_layer()
        # Measurement of all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear head that maps the measured amplitudes to a 4‑dimensional output
        self.head = nn.Linear(n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def _build_q_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
                self.attention = QuantumSelfAttention(n_wires)

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice) -> None:
                self.random_layer(qdev)
                self.attention(qdev)

        return QLayer(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that encodes the input image, refines it through a variational
        circuit with quantum self‑attention, and projects to a 4‑dimensional space.
        """
        bsz = x.shape[0]
        # Prepare a quantum device with batch support
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Average‑pool image to match the encoder input size
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Encode classical features into the quantum state
        self.encoder(qdev, pooled)
        # Apply the variational block
        self.q_layer(qdev)
        # Measure all qubits
        features = self.measure(qdev)
        # Linear head and normalization
        out = self.head(features)
        return self.norm(out)

__all__ = ["QuantumNATGen221"]
