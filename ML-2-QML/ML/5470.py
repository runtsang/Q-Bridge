"""Hybrid estimator that fuses classical CNN, quantum fully‑connected, and QCNN layers.

The module can be instantiated in three modes:
    * ``mode="classic"`` – pure feed‑forward neural network (inherits from EstimatorQNN).
    * ``mode="quantum"`` – classical backbone + torchquantum QFCModel as a quantum layer.
    * ``mode="hybrid"`` – classical backbone + QCNN‑style ansatz implemented as a parameterized circuit in the forward pass.

The design allows easy swapping of the quantum component without touching the training loop.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# --- Classical blocks -------------------------------------------------------
class EstimatorQNN(nn.Module):
    """Simple fully‑connected regression network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QCNNFeatureMap(nn.Module):
    """Feature map inspired by QCNN – a shallow CNN followed by a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --- Quantum block (torchquantum) --------------------------------------------
# We keep the QFCModel from QuantumNAT as a sub‑module.
# Import only if torchquantum is available; otherwise raise informative error.
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError as exc:
    tq = None
    tqf = None
    exc_msg = str(exc)

class QFCQuantumLayer(tq.QuantumModule if tq else nn.Module):
    """Quantum fully‑connected layer using torchquantum."""
    if tq:
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = 4
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

        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.q_layer = self.QLayer()
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(self.n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
            pooled = F.avg_pool2d(x, 6).view(bsz, 16)
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)
    else:
        def __init__(self, *_, **__):
            raise RuntimeError(f"torchquantum is required for QFCQuantumLayer: {exc_msg}")

# --- Hybrid estimator ---------------------------------------------------------
class HybridEstimatorQNN(nn.Module):
    """Composable estimator that can be classical, quantum, or hybrid."""
    def __init__(self, mode: str = "classic", use_qfc: bool = False) -> None:
        """
        Parameters
        ----------
        mode:
            * ``classic`` – only EstimatorQNN.
            * ``quantum`` – EstimatorQNN + QFCQuantumLayer.
            * ``hybrid``  – EstimatorQNN + QCNNFeatureMap + optional QFCQuantumLayer.
        use_qfc:
            If True and mode supports it, attach a quantum fully‑connected layer.
        """
        super().__init__()
        self.mode = mode
        self.base = EstimatorQNN()
        self.qfc = QFCQuantumLayer() if use_qfc else None
        self.qcnn = QCNNFeatureMap() if mode == "hybrid" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base feed‑forward
        out = self.base(x)

        # QCNN feature map if in hybrid mode
        if self.qcnn is not None:
            out = self.qcnn(out)

        # Quantum fully‑connected layer if enabled
        if self.qfc is not None:
            # Expectation value from the quantum layer
            quantum_out = self.qfc(out)
            # Concatenate classical and quantum outputs
            out = torch.cat([out, quantum_out], dim=-1)

        return out

__all__ = ["HybridEstimatorQNN"]
