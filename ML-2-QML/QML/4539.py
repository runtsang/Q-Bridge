import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter based on a two‑qubit kernel."""
    def __init__(self) -> None:
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

class KernalAnsatz(tq.QuantumModule):
    """Encodes two classical vectors via a fixed gate list."""
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

class QuantumKernelLayer(tq.QuantumModule):
    """Quantum kernel mapping to a trainable set of reference vectors."""
    def __init__(self, num_refs: int, input_dim: int) -> None:
        super().__init__()
        self.num_refs = num_refs
        self.input_dim = input_dim
        self.ref_vectors = nn.Parameter(torch.randn(num_refs, input_dim))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        outputs = []
        for i in range(self.num_refs):
            y = self.ref_vectors[i].unsqueeze(0).repeat(batch, 1)
            self.ansatz(self.q_device, x, y)
            outputs.append(self.q_device.states.view(-1)[0].unsqueeze(1))
        return torch.cat(outputs, dim=1)

class QuantumSelfAttention(tq.QuantumModule):
    """Variational self‑attention circuit producing a 4‑dimensional feature."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.rotation_params = nn.Parameter(torch.randn(n_qubits * 3))
        self.entangle_params = nn.Parameter(torch.randn(n_qubits - 1))
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        self.q_device.reset_states(batch)
        for i in range(self.n_qubits):
            self.q_device.rx(self.rotation_params[3 * i], wires=i)
            self.q_device.ry(self.rotation_params[3 * i + 1], wires=i)
            self.q_device.rz(self.rotation_params[3 * i + 2], wires=i)
        for i in range(self.n_qubits - 1):
            self.q_device.crx(self.entangle_params[i], wires=[i, i + 1])
        return self.measure(self.q_device)

class HybridQuantumNat(tq.QuantumModule):
    """Hybrid quantum model combining quanvolution, quantum kernel and variational attention."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.kernel_layer = QuantumKernelLayer(num_refs=64, input_dim=4 * 14 * 14)
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.classifier = nn.Linear(4 * 14 * 14 + 64 + 4, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qfilter_feat = self.qfilter(x)
        kernel_feat = self.kernel_layer(qfilter_feat)
        attn_feat = self.attention(x)
        combined = torch.cat([qfilter_feat, kernel_feat, attn_feat], dim=1)
        logits = self.classifier(combined)
        return self.norm(logits)

__all__ = ["HybridQuantumNat"]
