import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolutional filter as in the original quanvolution example."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
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
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class EstimatorQNN(tq.QuantumModule):
    """Simple quantum regressor using a single qubit."""
    def __init__(self):
        super().__init__()
        self.n_wires = 1
        self.weight = nn.Parameter(torch.randn(1))
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [0], "func": "ry", "wires": [0]}]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(q_device, x)
        tq.ry(q_device, self.weight.repeat(x.shape[0]), wires=[0])
        measurement = self.measure(q_device)
        return measurement.view(-1, 1)

class SamplerQNN(tq.QuantumModule):
    """Quantum sampler with two qubits and trainable Ry gates."""
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.input_encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.weights = nn.Parameter(torch.randn(4))
        self.cx1 = tq.CX()
        self.cx2 = tq.CX()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.input_encoder(q_device, x)
        self.cx1(q_device, wires=[0, 1])
        for i, w in enumerate(self.weights):
            tq.ry(q_device, w, wires=[i % 2])
        self.cx2(q_device, wires=[0, 1])
        measurement = self.measure(q_device)
        probs = torch.softmax(measurement, dim=-1)
        return probs

class HybridQuanvolution(nn.Module):
    """Quantum-enhanced hybrid classifier that integrates a classical quanvolution filter,
    a quantum kernel, and a linear classifier."""
    def __init__(self, num_classes: int = 10, num_prototypes: int = 20, feature_dim: int = 4 * 14 * 14):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.kernel = QuantumKernel()
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        k_values = []
        for proto in self.prototypes:
            k_val = self.kernel(features, proto.unsqueeze(0))
            k_values.append(k_val)
        k_values = torch.cat(k_values, dim=1)
        logits = self.classifier(k_values)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuantumKernel", "EstimatorQNN", "SamplerQNN", "HybridQuanvolution"]
