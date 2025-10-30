import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class ConvGen340(tq.QuantumModule):
    """Hybrid quantum‑classical convolutional backbone.

    It mirrors the classical ConvGen340 but replaces the 2‑D filter
    with a variational circuit that processes a flattened patch of the
    feature map.  The circuit is built from torchquantum layers and
    is trainable via back‑propagation.
    """

    class QLayer(tq.QuantumModule):
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
                 kernel_size: int = 2,
                 stride: int = 1,
                 threshold: float = 0.0,
                 device: str = "cpu"):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold

        # Classical feature extractor (same as in the ML version)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Quantum filter
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Extract classical features
        feat = self.features(x)
        # Flatten to a vector and take the first n_wires elements
        pooled = feat.view(bsz, -1)[:, :self.n_wires]
        # Encode into quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that accepts a 2‑D tensor and returns a tensor."""
        return self.forward(data)

__all__ = ["ConvGen340"]
