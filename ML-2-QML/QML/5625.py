"""Quantum sampler that fuses a quantum LSTM, quantum kernel, and a parameterized circuit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torchquantum.functional as tqf

# --- Quantum LSTM cell ------------------------------------
class QLSTMQuantum(tq.QuantumModule):
    """LSTM where each gate is a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tq.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states=None):
        # inputs: (batch, seq_len, feature_dim)
        seq_len = inputs.size(1)
        inputs_t = inputs.transpose(0, 1)  # (seq_len, batch, feature_dim)
        hx, cx = self._init_states(inputs_t, states)
        outputs = []
        for x in inputs_t.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
        return outputs.transpose(0, 1), (hx, cx)

    def _init_states(self, inputs_t, states):
        if states is not None:
            return states
        batch_size = inputs_t.size(1)
        device = inputs_t.device
        hidden = torch.zeros(batch_size, self.linear_forget.out_features, device=device)
        cell = torch.zeros(batch_size, self.linear_forget.out_features, device=device)
        return hidden, cell

# --- Quantum kernel ------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes two classical vectors into a quantum state and applies a symmetric circuit."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class KernelQuantum(tq.QuantumModule):
    """Quantum kernel that returns the absolute overlap of two encoded states."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# --- Hybrid quantum sampler ------------------------------------
class SamplerQNN(tq.QuantumModule):
    """Combines a quantum LSTM, quantum kernel, and a parameterized sampler circuit."""
    def __init__(self, n_qubits_lstm: int = 8, n_qubits_sampler: int = 4):
        super().__init__()
        self.lstm = QLSTMQuantum(input_dim=n_qubits_lstm, hidden_dim=n_qubits_lstm, n_qubits=n_qubits_lstm)
        self.kernel = KernelQuantum()
        self.n_qubits_sampler = n_qubits_sampler
        # Reference vector for kernel similarity
        self.ref_vector = torch.randn(1, n_qubits_lstm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim) classical input sequence.
        Returns:
            probs: (batch, 2**n_qubits_sampler) probability distribution over sampler outputs.
        """
        # Quantum LSTM to obtain hidden representations
        lstm_out, _ = self.lstm(x)  # shape (batch, seq_len, hidden_dim)
        # Kernel similarity between mean hidden state and reference
        sim = self.kernel(lstm_out.mean(dim=1), self.ref_vector).unsqueeze(-1)  # (batch,1)
        # Parameterized sampler circuit
        qdev = tq.QuantumDevice(n_wires=self.n_qubits_sampler, bsz=x.size(0), device=x.device)
        # Apply ry gates driven by the first `n_qubits_sampler` components of the hidden state
        hidden_mean = lstm_out.mean(dim=1)  # (batch, hidden_dim)
        for i in range(self.n_qubits_sampler):
            param = hidden_mean[:, i] if hidden_mean.size(1) > i else torch.zeros(x.size(0), device=x.device)
            tq.ry(qdev, wires=[i], params=param)
        # Entangle qubits with a ring of CNOTs
        for i in range(self.n_qubits_sampler - 1):
            tq.cnot(qdev, wires=[i, i + 1])
        tq.cnot(qdev, wires=[self.n_qubits_sampler - 1, 0])
        # Measure in computational basis
        probs = tq.measure(qdev, measure_type="mps")  # shape (batch, 2**n_qubits_sampler)
        # Weight by kernel similarity
        probs = probs * sim
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

__all__ = ["SamplerQNN"]
