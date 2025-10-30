import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class EstimatorQNNHybrid(tq.QuantumModule):
    # Hybrid quantum‑classical estimator that mirrors the classical model above.
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3)
            tqf.sx(qdev, wires=2)
            tqf.cnot(qdev, wires=[3, 0])

    def __init__(self, input_params, layers) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.reduce = nn.Linear(16 * 7 * 7, 4)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.final_fc = nn.Linear(4, 1)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.cnn(x)
        flattened = features.view(bsz, -1)
        reduced = self.reduce(flattened)
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=x.device, record_op=True)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        out = self.final_fc(out)
        return out

def EstimatorQNN() -> EstimatorQNNHybrid:
    # Convenience factory that reproduces the original EstimatorQNN signature.
    return EstimatorQNNHybrid(None, None)

__all__ = ["EstimatorQNNHybrid", "EstimatorQNN"]
