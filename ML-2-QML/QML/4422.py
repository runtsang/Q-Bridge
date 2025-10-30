"""Hybrid convolutional network with quantum kernel and NAT modules.

This module defines HybridConvNet, a drop‑in replacement for the
original hybrid quantum network.  It combines a classical convolutional
backbone, a quantum kernel (TorchQuantum ansatz), a quantum NAT module
(QFCModel with QLayer and encoder), and a quantum expectation head
implemented with Qiskit.  The architecture is intentionally symmetrical
to the classical version in ml_code, providing a direct comparison of
quantum vs classical implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer, execute, assemble, transpile
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum kernel module (TorchQuantum)
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data into a quantum circuit."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            tqf.__dict__[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            tqf.__dict__[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
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

# Quantum NAT module (TorchQuantum)
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

class QFCModel(tq.QuantumModule):
    """Quantum fully connected model inspired by the Quantum‑NAT paper."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer()
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

# Quantum expectation head using Qiskit
class QuantumExpectationHead:
    """Variational circuit that returns the probability of measuring |1>."""
    def __init__(self, backend, shots: int, shift: float):
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = []
        for val in inputs.tolist():
            param_bind = {self.theta: float(val) + self.shift}
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_bind])
            result = job.result().get_counts(self.circuit)
            # probability of measuring |1>
            prob_1 = sum(count / self.shots for state, count in result.items() if state == '1')
            probs.append(prob_1)
        return torch.tensor(probs).unsqueeze(-1)

class HybridConvNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55819, 120)  # 55815 conv features + 4 NAT features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum kernel and NAT modules
        self.kernel = Kernel()
        self.nat = QFCModel()

        # Quantum expectation head
        backend = Aer.get_backend("qasm_simulator")
        self.hybrid = QuantumExpectationHead(backend=backend, shots=100, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        nat_features = self.nat(inputs[:, :1, :, :])  # use first channel for NAT
        combined = torch.cat([x, nat_features], dim=1)

        x = F.relu(self.fc1(combined))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        probs = self.hybrid(x.squeeze(-1))
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridConvNet"]
