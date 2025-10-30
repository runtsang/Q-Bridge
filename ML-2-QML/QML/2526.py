import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict

class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel module: encodes two vectors into a quantum state and returns the overlap.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoding_ops = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(1)
        # Encode x
        for info in self.encoding_ops:
            func = func_name_dict[info["func"]]
            params = x[:, info["input_idx"]] if func.num_params else None
            func(self.q_device, wires=info["wires"], params=params)
        # Encode y with negative parameters
        for info in reversed(self.encoding_ops):
            func = func_name_dict[info["func"]]
            params = -y[:, info["input_idx"]] if func.num_params else None
            func(self.q_device, wires=info["wires"], params=params)
        return torch.abs(self.q_device.states[0])

class HybridQuantumNATQuantum(tq.QuantumModule):
    """
    Quantum hybrid model: quantum encoder → variational quantum layer → measurement.
    Optional quantum kernel augmentation mirrors the classical counterpart.
    """
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

    def __init__(self, use_kernel: bool = False, num_classes: int = 4):
        super().__init__()
        self.use_kernel = use_kernel
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        if use_kernel:
            self.kernel = QuantumKernel(self.n_wires)
            self.prototype = nn.Parameter(torch.randn(10, self.n_wires), requires_grad=True)
            self.classifier = nn.Linear(10, num_classes)
        else:
            self.classifier = nn.Linear(self.n_wires, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)

        if self.use_kernel:
            k = torch.zeros(bsz, 10, device=x.device)
            for i in range(bsz):
                for j in range(10):
                    k[i, j] = self.kernel(out[i], self.prototype[j])
            logits = self.classifier(k)
        else:
            logits = self.classifier(out)
        return logits

__all__ = ["HybridQuantumNATQuantum"]
