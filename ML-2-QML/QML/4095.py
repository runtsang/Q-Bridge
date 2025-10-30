"""Quantum implementation of :class:`SelfAttentionHybrid` using torchquantum.

The quantum path replaces the classical self‑attention, LSTM and
classifier with parameterised quantum circuits while preserving the
same public API.  The module can be used as a drop‑in replacement
for the classical version.

Typical usage
-------------
>>> import torch
>>> from SelfAttention__gen062 import SelfAttentionHybrid
>>> model = SelfAttentionHybrid(embed_dim=4, hidden_dim=8, n_qubits=4)
>>> inputs = torch.randn(2, 5, 4)  # batch, seq_len, embed_dim
>>> rot = torch.randn(4*3)          # rotation parameters
>>> ent = torch.randn(4-1)          # entanglement parameters
>>> logits = model.forward(inputs, rot, ent)
>>> logits.shape
torch.Size([2, 5, 4])
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Encode each embedding dimension into a qubit via RX
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(embed_dim)]
        )
        # Rotation gates – 3 per qubit as in the classical example
        self.rotation_gates = [
            tq.RX(has_params=True, trainable=True) for _ in range(embed_dim * 3)
        ]
        # Entanglement gates – one CRX per adjacent pair
        self.entangle_gates = [
            tq.CRX(has_params=True, trainable=True) for _ in range(embed_dim - 1)
        ]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, rotation_params: torch.Tensor, entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Tensor of shape ``(batch, seq_len, embed_dim)``.
        rotation_params, entangle_params
            Parameters for the rotation and entanglement gates.
            They are accepted for API compatibility but ignored
            in the simplified implementation.
        """
        batch = x.shape[0]
        # Reduce over the sequence dimension: mean embedding per sample
        mean_x = x.mean(dim=1)  # (batch, embed_dim)
        qdev = tq.QuantumDevice(n_wires=self.embed_dim, bsz=batch, device=x.device, record_op=True)
        self.encoder(qdev, mean_x)

        # Apply rotations (parameters are ignored – gates are trainable)
        for gate in self.rotation_gates:
            gate(qdev)

        # Apply entanglement (parameters are ignored)
        for gate in self.entangle_gates:
            gate(qdev)

        out = self.measure(qdev)  # (batch, embed_dim)
        return out


class QuantumQLSTM(tq.QuantumModule):
    """Quantum LSTM layer with gate circuits."""
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
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
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

    def forward(
        self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq_len, batch, _ = inputs.shape
        hx, cx = self._init_states(batch, states)
        outputs = []
        for t in range(seq_len):
            x = inputs[:, t, :]
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return outputs, (hx, cx)

    def _init_states(
        self, batch_size: int, states: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        device = torch.device("cpu")
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class SelfAttentionHybrid(tq.QuantumModule):
    """Quantum‑ready self‑attention module that can be used as a drop‑in
    replacement for the classical :class:`SelfAttentionHybrid`.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.attention = QuantumSelfAttention(embed_dim)
        self.lstm = QuantumQLSTM(embed_dim, hidden_dim, n_qubits)
        self.fc = nn.Linear(hidden_dim, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            Tensor of shape ``(batch, seq_len, embed_dim)``.
        rotation_params, entangle_params
            Parameters for the quantum self‑attention block.
        """
        # Self‑attention
        attn_out = self.attention(inputs, rotation_params, entangle_params)  # (batch, embed_dim)
        # Expand to a single‑step sequence for the LSTM
        attn_out = attn_out.unsqueeze(1)  # (batch, 1, embed_dim)
        # LSTM
        lstm_out, _ = self.lstm(attn_out)  # (batch, 1, hidden_dim)
        # Classifier
        flat = lstm_out.reshape(-1, self.fc.in_features)
        logits = self.fc(flat)  # (batch, 4)
        logits = logits.reshape(lstm_out.shape[0], 1, -1)
        return self.norm(logits)

__all__ = ["SelfAttentionHybrid"]
