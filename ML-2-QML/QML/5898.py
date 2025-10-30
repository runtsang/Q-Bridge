import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter producing 4 features per image."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        flat = x.view(bsz, -1)[:, :4]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, flat)
        self.q_layer(qdev)
        measurement = self.measure(qdev)  # shape [batch, 4]
        return measurement

class QuantumSampler(tq.QuantumModule):
    """Quantum sampler mapping 4 input features to class logits."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        logits = self.linear(measurement)
        return F.log_softmax(logits, dim=-1)

class HybridQuanvolutionSamplerClassifier(nn.Module):
    """Hybrid quantum-classical classifier: quantum quanvolution filter + quantum sampler."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        self.sampler = QuantumSampler(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)  # shape [batch, 4]
        logits = self.sampler(features)
        return logits

__all__ = ["HybridQuanvolutionSamplerClassifier"]
