from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum analogue of a 2×2 image patch extractor.  Each patch is
    encoded into a 4‑wire device, processed by a random circuit, and
    measured in the Pauli‑Z basis.
    """
    def __init__(self, channels: int = 4) -> None:
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
        self.random = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
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
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, data)
                self.random(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumFCL(tq.QuantumModule):
    """
    Parameterised quantum circuit that emulates a fully connected
    operation.  It accepts a vector of angles and returns a
    single expectation value.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        vals = thetas.view(-1, 1).float()
        expectation = torch.tanh(self.linear(vals)).mean(dim=0)
        return expectation

class QuantumQLSTM(tq.QuantumModule):
    """
    Quantum‑enhanced LSTM cell where each gate is realised by a
    small parameterised quantum circuit.
    """
    class _Gate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, n_wires: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.hidden_dim = hidden_dim
        self.forget_gate = self._Gate(n_wires)
        self.input_gate = self._Gate(n_wires)
        self.update_gate = self._Gate(n_wires)
        self.output_gate = self._Gate(n_wires)
        self.linear_f = nn.Linear(n_wires + hidden_dim, n_wires)
        self.linear_i = nn.Linear(n_wires + hidden_dim, n_wires)
        self.linear_u = nn.Linear(n_wires + hidden_dim, n_wires)
        self.linear_o = nn.Linear(n_wires + hidden_dim, n_wires)

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_f(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_i(combined)))
            u = torch.tanh(self.update_gate(self.linear_u(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_o(combined)))
            cx = f * cx + i * u
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QuantumClassifierCircuit(tq.QuantumModule):
    """
    Variational circuit that encodes a feature vector, applies a
    depth‑controlled ansatz, and measures Pauli‑Z on each qubit.
    """
    def __init__(self, num_qubits: int, depth: int) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(num_qubits)
            ]
        )
        self.ansatz = nn.ModuleList()
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.ansatz.append(tq.RY(has_params=True, trainable=True))
            for qubit in range(num_qubits - 1):
                self.ansatz.append(tq.CZ(wires=[qubit, qubit + 1]))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=x.shape[0], device=x.device)
        self.encoding(qdev, x)
        for gate in self.ansatz:
            gate(qdev)
        return self.measure(qdev)

class HybridClassifier(nn.Module):
    """
    Quantum‑hybrid classifier that stitches together the quanvolution
    filter, the fully‑connected quantum layer, the quantum LSTM and the
    variational classifier circuit.  All components are trainable
    within a single backward pass.
    """
    def __init__(self,
                 num_qubits: int,
                 lstm_hidden: int,
                 fcl_dim: int,
                 quanv_channels: int,
                 depth: int = 2) -> None:
        super().__init__()
        self.quanv = QuantumQuanvolutionFilter(quanv_channels)
        self.fcl = QuantumFCL(fcl_dim)
        self.lstm = QuantumQLSTM(num_qubits, lstm_hidden)
        self.classifier = QuantumClassifierCircuit(num_qubits, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quanv(x)
        x = self.fcl(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        logits = self.classifier(x.squeeze(1))
        return torch.log_softmax(logits, dim=-1)

__all__ = ["HybridClassifier"]
