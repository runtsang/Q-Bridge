import torch
import torch.nn as nn
import torchquantum as tq

class QuantumNATGen167(tq.QuantumModule):
    """Quantum component of the hybrid CNN‑QNN.

    The module accepts a 1‑D feature vector per batch item, encodes it
    using a 4‑qubit GeneralEncoder, applies a depth‑50 RandomLayer,
    a set of trainable single‑qubit gates, and concludes with a
    Hadamard–SX–CNOT sequence. The expectation of Pauli‑Z on all
    qubits is returned, followed by batch‑normalisation.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tq.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tq.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tq.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        bsz = features.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=features.device,
            record_op=True,
        )
        self.encoder(qdev, features)
        self.forward(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATGen167"]
