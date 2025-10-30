import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridQuantumNatNet(tq.QuantumModule):
    """
    Hybrid classical‑quantum network that merges Quantum‑NAT style CNNs
    with a variational quantum circuit and a hybrid head.
    """

    class QLayer(tq.QuantumModule):
        """
        Variational quantum layer with a random circuit and a few
        parameterised gates that act on 4 wires.
        """
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self,
                 in_channels: int = 1,
                 n_qubits: int = 4,
                 shift: float = 0.0,
                 device: str = "cpu"):
        super().__init__()
        # CNN backbone (identical to the classical version)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

        # Quantum block
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that uses the quantum circuit to produce a
        parameter‑shift style expectation value, then combines it
        with the classical projection.
        """
        bsz = x.shape[0]
        features = self.backbone(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)

        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                bsz=bsz,
                                device=x.device,
                                record_op=True)

        # Encode classical features into quantum state
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Apply variational layer
        self.q_layer(qdev)

        # Measure expectation values
        qout = self.measure(qdev)

        # Combine classical and quantum outputs
        combined = out + qout

        # Parameter‑shift style sigmoid with optional shift
        probs = torch.sigmoid(combined + self.shift)
        return probs

__all__ = ["HybridQuantumNatNet"]
