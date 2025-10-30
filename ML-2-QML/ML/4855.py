import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class _FeatureExtractor(nn.Module):
    """Light‑weight CNN that produces 16‑dim feature vectors."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))

class _EstimatorNN(nn.Module):
    """Tiny regression head used after the quantum layer."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _SamplerModule(nn.Module):
    """Simple sampler head that produces class probabilities."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class HybridQuantumBlock(tq.QuantumModule):
    """Quantum core that encodes a 16‑dim vector into 4 qubits and applies a variational layer."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.layer = self._QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    class _QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.cz = tq.CZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            self.cz(qdev, wires=[1, 3])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, x)
        self.layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class HybridNatEstimatorSampler(nn.Module):
    """Combined hybrid model with CNN, quantum block, regression and sampler heads."""
    def __init__(self) -> None:
        super().__init__()
        self.features = _FeatureExtractor()
        self.quantum = HybridQuantumBlock()
        self.estimator = _EstimatorNN()
        self.sampler = _SamplerModule()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Classical feature extraction
        feat = self.features(x)
        # Reduce spatial dimension
        pooled = F.avg_pool2d(feat, kernel_size=6).view(x.shape[0], -1)
        # Quantum encoding & variational layer
        q_vec = self.quantum(pooled)
        # Dual heads
        reg = self.estimator(q_vec)
        cls = self.sampler(q_vec)
        return {"regression": reg, "classification": cls}

__all__ = ["HybridNatEstimatorSampler"]
