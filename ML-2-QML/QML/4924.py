import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block using RX, RY, RZ and CRX gates."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        # Trainable rotation parameters for each qubit
        self.rotation_params = tq.ParameterList([
            tq.Parameter() for _ in range(n_qubits * 3)
        ])
        # Trainable entanglement parameters between neighbours
        self.entangle_params = tq.ParameterList([
            tq.Parameter() for _ in range(n_qubits - 1)
        ])

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        # Apply rotations
        for i in range(self.n_qubits):
            idx = i * 3
            qdev.rx(self.rotation_params[idx], i)
            qdev.ry(self.rotation_params[idx + 1], i)
            qdev.rz(self.rotation_params[idx + 2], i)
        # Apply entanglement
        for i in range(self.n_qubits - 1):
            qdev.crx(self.entangle_params[i], i, i + 1)
        # Return the device state for downstream use
        return qdev.states

class KernalAnsatz(tq.QuantumModule):
    """Fixed ansatz that encodes two classical vectors into a quantum state."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor):
        qdev.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](qdev, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](qdev, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel that returns the absolute overlap of two encoded states."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.qdev, x, y)
        return torch.abs(self.qdev.states.view(-1)[0])

class HybridNATModel(tq.QuantumModule):
    """
    Quantum counterpart of HybridNATModel:
    * General encoder extracts classical features into a quantum device
    * QLayer implements a random circuit with trainable single‑qubit gates
    * Kernel evaluates a quantum RBF‑style similarity
    * QuantumSelfAttention adds a parametric attention block
    * Final measurement and batch‑norm produce the output
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.kernel = Kernel()
        self.attention = QuantumSelfAttention()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=bsz,
                                device=x.device,
                                record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)

        # Kernel similarity with a fixed prototype
        prototype = torch.randn(1, 16, device=x.device)
        kernel_sim = self.kernel(pooled, prototype.repeat(bsz, 1))

        # Quantum self‑attention
        self.attention(qdev)

        out = self.measure(qdev)
        # Concatenate kernel similarity to the measurement output
        out = torch.cat([out, kernel_sim], dim=-1)
        return self.norm(out)

__all__ = ["HybridNATModel"]
