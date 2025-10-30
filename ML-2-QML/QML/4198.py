import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import op_name_dict, func_name_dict

class QuantumKernalAnsatz(tq.QuantumModule):
    """Encodes classical data into a quantum state via a list of gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluating similarity via a fixed ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def _kernel_single(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sims = []
        for i in range(x.shape[0]):
            sims.append(self._kernel_single(x[i:i+1], y))
        return torch.cat(sims, dim=0)

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter applying a random two‑qubit circuit to 2x2 patches."""
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
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumQLayer(tq.QuantumModule):
    """Parameterized layer used within the quantum fully‑connected backbone."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.random_layer(q_device)
        self.rx0(q_device, wires=0)
        self.ry0(q_device, wires=1)
        self.rz0(q_device, wires=3)
        self.crx0(q_device, wires=[0, 2])
        tqf.hadamard(q_device, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(q_device, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(q_device, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

class QuantumQFCBackbone(tq.QuantumModule):
    """Quantum version of the convolution‑fully‑connected encoder."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QuantumQLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class HybridQuantumNAT(tq.QuantumModule):
    """Hybrid quantum model combining quanvolution, quantum kernel, and a quantum fully‑connected backbone."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.quanv = QuantumQuanvolutionFilter()
        self.kernel = QuantumKernel()
        self.backbone = QuantumQFCBackbone()
        self.linear = nn.Linear(4 + 1, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)
        self.prototype = nn.Parameter(torch.zeros(1, 4 * 14 * 14))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quanv_features = self.quanv(x)
        sim = self.kernel(quanv_features, self.prototype)
        backbone_out = self.backbone(x)
        combined = torch.cat([backbone_out, sim], dim=1)
        logits = self.linear(combined)
        return self.norm(logits)

__all__ = ["HybridQuantumNAT"]
