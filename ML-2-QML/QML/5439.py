from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict, op_name_dict

class EstimatorQNNQuantum(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 1
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.params = tq.RX(has_params=True, trainable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        self.params(self.q_device, wires=0, params=x[:,0])
        self.q_device.measure_all()
        return self.q_device.states.real

class QLSTMQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class SamplerQNNQuantum(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.params0 = tq.RX(has_params=True, trainable=True)
        self.params1 = tq.RX(has_params=True, trainable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        self.q_device.ry(x[:,0], 0)
        self.q_device.ry(x[:,1], 1)
        self.q_device.cx(0,1)
        self.params0(self.q_device, wires=0, params=x[:,2])
        self.params1(self.q_device, wires=1, params=x[:,3])
        self.q_device.cx(0,1)
        self.q_device.measure_all()
        return self.q_device.states.real

class KernalAnsatzQuantum(tq.QuantumModule):
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

class KernelQuantum(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatzQuantum(
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = KernelQuantum()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def EstimatorQNNQuantumModel(mode: str = "regression", **kwargs):
    """Return a quantum EstimatorQNN model in the requested mode."""
    if mode == "regression":
        return EstimatorQNNQuantum()
    elif mode == "tagger":
        return QLSTMQuantum(**kwargs)
    elif mode == "sampler":
        return SamplerQNNQuantum()
    elif mode == "kernel":
        return KernelQuantum()
    else:
        raise ValueError(f"Unsupported mode {mode}")

__all__ = [
    "EstimatorQNNQuantumModel",
    "EstimatorQNNQuantum",
    "QLSTMQuantum",
    "SamplerQNNQuantum",
    "KernelQuantum",
    "kernel_matrix",
]
