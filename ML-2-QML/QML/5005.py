from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNATModel(tq.QuantumModule):
    """
    Quantum‑augmented version of HybridNATModel.

    Features:
      * Classical preprocessing via a small CNN.
      * Encoding of the flattened features into a 4‑qubit device
        using a GeneralEncoder.
      * A variational layer consisting of a random layer followed by
        trainable RX/RY/RZ/CRX gates, mirroring the QFCModel.
      * Optional weight clipping and learnable scale/shift.
      * Task‑specific output head (softmax or linear regression).
    """

    class QLayer(tq.QuantumModule):
        """Variational block with a random layer and trainable rotations."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(
        self,
        num_channels: int = 1,
        conv_out_channels: int = 16,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        n_wires: int = 4,
        clip_weights: bool = False,
        clip_bound: float = 5.0,
        task: str = "classification",
        num_classes: int = 2,
        regression_dim: int = 1,
    ) -> None:
        super().__init__()

        self.clip_weights = clip_weights
        self.clip_bound = clip_bound

        # Classical feature extractor (identical to ML version)
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, conv_out_channels, kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dummy pass to compute flattened size (same as ML)
        dummy_input = torch.zeros(1, num_channels, 28, 28)
        with torch.no_grad():
            feat = self.features(dummy_input)
        flat_size = feat.numel()

        # Encoder and variational layer
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Learnable scaling & shift
        self.register_buffer("scale", torch.ones(self.n_wires))
        self.register_buffer("shift", torch.zeros(self.n_wires))

        # Task head
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(self.n_wires, num_classes),
                nn.LogSoftmax(dim=1),
            )
        elif task == "regression":
            self.head = nn.Linear(self.n_wires, regression_dim)
        else:
            raise ValueError("task must be 'classification' or'regression'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Classical feature extraction
        feat = self.features(x)
        pooled = F.avg_pool2d(feat, 6).view(bsz, -1)  # match QFCModel's pooling

        # Quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode classical features
        self.encoder(qdev, pooled)

        # Variational layer
        self.q_layer(qdev)

        # Measurement
        out = self.measure(qdev)

        # Optional weight clipping (on variational gates)
        if self.clip_weights:
            with torch.no_grad():
                for param in self.q_layer.parameters():
                    param.clamp_(-self.clip_bound, self.clip_bound)

        # Normalization
        out = self.norm(out)

        # Scaling & shift
        out = out * self.scale + self.shift

        # Head
        return self.head(out)

__all__ = ["HybridNATModel"]
