import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumRBFKernel(tq.QuantumModule):
    """Quantum version of the RBF kernel using a parameterized ansatz."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Fixed ansatz that evaluates a quantum kernel."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumRBFKernel([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumEstimatorQNN(tq.QuantumModule):
    """Quantum regression unit inspired by EstimatorQNN."""
    def __init__(self):
        super().__init__()
        self.n_wires = 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [0]},  # weight
        ])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder(self.q_device, x)
        measurement = self.measure(self.q_device)
        return measurement.mean()

class HybridQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that maps 2×2 image patches to a 4‑dimensional feature vector."""
    def __init__(self, kernel_size: int = 2, stride: int = 2, n_wires: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, self.kernel_size):
            for c in range(0, 28, self.kernel_size):
                data = torch.stack([
                    x[:, r, c],
                    x[:, r, c + 1],
                    x[:, r + 1, c],
                    x[:, r + 1, c + 1],
                ], dim=1)
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class HybridQuanvolutionClassifier(nn.Module):
    """Hybrid quantum‑classical classifier that uses a quantum filter and a linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = HybridQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

class HybridQuanvolutionRegressor(nn.Module):
    """Quantum‑classical regressor that couples the filter with a quantum estimator."""
    def __init__(self):
        super().__init__()
        self.qfilter = HybridQuanvolutionFilter()
        self.estimator = QuantumEstimatorQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x).view(x.size(0), -1)
        return self.estimator(features)

__all__ = [
    "QuantumRBFKernel",
    "QuantumKernel",
    "QuantumEstimatorQNN",
    "HybridQuanvolutionFilter",
    "HybridQuanvolutionClassifier",
    "HybridQuanvolutionRegressor",
]
