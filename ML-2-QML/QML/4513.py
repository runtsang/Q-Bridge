"""Hybrid quantum‑classical estimator combining QCNN feature extraction and a
Quantum‑NAT style quantum layer.  The model is a torchquantum.QuantumModule
and can be trained with standard PyTorch optimizers via the quantum backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridEstimatorQNN(tq.QuantumModule):
    """
    Quantum‑classical hybrid estimator.

    Parameters
    ----------
    num_features : int, default 2
        Number of classical input features.
    num_classes : int, default 1
        If >1 the network outputs a probability distribution via softmax.
    """

    class QLayer(tq.QuantumModule):
        """Quantum layer inspired by Quantum‑NAT.

        Includes a random layer followed by deterministic rotations and a
        small entangling block.  The design mirrors the QLayer in the
        QuantumNAT example but is compact enough for 4 wires.
        """

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
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

    def __init__(self, num_features: int = 2, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Classical feature extractor – QCNN‑style
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, 8), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Quantum block
        self.q_layer = self.QLayer()

        # Measurement and classical read‑out
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.q_layer.n_wires)

        # Final head
        if self.num_classes == 1:
            self.head = nn.Linear(self.q_layer.n_wires, 1)
        else:
            self.head = nn.Linear(self.q_layer.n_wires, self.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical feature extraction
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode classical features into quantum state
        # Here we use a simple linear mapping to the first two wires
        tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
        tqf.hadamard(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
        tqf.rx(qdev, x[:, 0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tqf.ry(qdev, x[:, 1], wires=1, static=self.static_mode, parent_graph=self.graph)

        # Apply quantum layer
        self.q_layer(qdev)

        # Measurement
        out = self.measure(qdev)
        out = self.norm(out)

        # Classical head
        out = self.head(out)
        if self.num_classes > 1:
            out = F.softmax(out, dim=-1)
        return out


def HybridEstimatorQNNFactory(num_features: int = 2, num_classes: int = 1) -> HybridEstimatorQNN:
    """Factory returning a configured :class:`HybridEstimatorQNN`."""
    return HybridEstimatorQNN(num_features=num_features, num_classes=num_classes)


__all__ = ["HybridEstimatorQNN", "HybridEstimatorQNNFactory"]
