import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn as nn

class QuantumAttention(tq.QuantumModule):
    """Quantum self‑attention block using a simple parameterised circuit."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: torch.Tensor) -> torch.Tensor:
        seq_len, batch, embed_dim = inputs.shape
        outputs = []
        for t in range(seq_len):
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=inputs.device)
            for i in range(self.n_qubits):
                angle_rx = rotation_params[3*i] * inputs[t, :, i]
                angle_ry = rotation_params[3*i+1] * inputs[t, :, i]
                angle_rz = rotation_params[3*i+2] * inputs[t, :, i]
                tq.RX(angle_rx, wires=[i])(qdev)
                tq.RY(angle_ry, wires=[i])(qdev)
                tq.RZ(angle_rz, wires=[i])(qdev)
            for i in range(self.n_qubits - 1):
                tq.CRZ(entangle_params[i], control=i, target=i+1)(qdev)
            out = tq.MeasureAll(tq.PauliZ)(qdev)
            outputs.append(out)
        return torch.stack(outputs, dim=0)

class QuantumQLSTM(tq.QuantumModule):
    """Quantum‑enhanced LSTM cell (copy of the seed implementation)."""
    class QLayer(tq.QuantumModule):
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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class UnifiedSelfAttentionLSTM(tq.QuantumModule):
    """Combined quantum self‑attention + LSTM module."""
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attention = QuantumAttention(embed_dim)
        self.lstm = QuantumQLSTM(embed_dim, hidden_dim, n_qubits=hidden_dim)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out = self.attention(rotation_params, entangle_params, inputs)
        lstm_out, (hx, cx) = self.lstm(attn_out, states)
        return lstm_out, (hx, cx)

__all__ = ["UnifiedSelfAttentionLSTM"]
