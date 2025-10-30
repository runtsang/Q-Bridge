import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class FraudDetectionHybrid(tq.QuantumModule):
    """
    Quantum fraud detection model.
    Encodes classical features into qubits, applies a random variational layer,
    measures in Pauli‑Z basis, and projects to a binary output with scale/shift buffers.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
            # Entanglement block
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1], static=False, parent_graph=self.graph)
            self.crx(qdev, wires=[0, self.n_wires - 1])

    def __init__(self, num_features: int, n_wires: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_features = num_features
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, num_classes)
        self.register_buffer("scale", torch.ones(num_classes))
        self.register_buffer("shift", torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: tensor of shape (batch, num_features)
        Returns logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        features = self.measure(qdev)
        logits = self.head(features)
        return logits * self.scale + self.shift

__all__ = ["FraudDetectionHybrid"]
