"""
Quantum modules used by `QuantumHybridModel`.  They rely on `torchquantum`
for variational circuits and are fully compatible with the classical
wrapper.  The modules implement a quantum self‑attention block and a
quantum LSTM cell that operate on dense feature vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np


class QuantumAttention(tq.QuantumModule):
    """
    Variational quantum circuit that receives a classical feature vector,
    encodes it with a general encoder, applies trainable RX gates,
    and measures in the computational basis.  The output is a 4‑dim
    probability vector that is used as a quantum attention embedding.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Encoder that maps each feature to a rotation on a separate qubit
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{n_qubits}x{n_qubits}_ryzxy"]
        )
        # Trainable rotation layer
        self.rxs = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, D) feature vector, where D >= n_qubits.
        Returns:
            (B, n_qubits) probability vector from measuring all qubits.
        """
        B, D = feats.shape
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=B, device=feats.device, record_op=True)
        # Use first n_qubits of the feature vector for encoding
        self.encoder(qdev, feats[:, :self.n_qubits])
        # Apply trainable rotations
        for gate in self.rxs:
            gate(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        # Convert counts to probabilities
        probs = out[:, 1] / out.sum(dim=1)
        return probs


class QuantumLSTMCell(tq.QuantumModule):
    """
    Quantum‑enhanced LSTM cell where each gate is a small variational circuit.
    The cell receives a feature vector and a hidden state, and outputs
    a new hidden state of the same dimensionality.
    """
    class _Gate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{n_wires}x{n_wires}_ryzxy"]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            # Simple CNOT chain to entangle
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits if n_qubits is not None else hidden_dim

        # Gates
        self.forget_gate = self._Gate(self.n_qubits)
        self.input_gate  = self._Gate(self.n_qubits)
        self.update_gate = self._Gate(self.n_qubits)
        self.output_gate = self._Gate(self.n_qubits)

        # Linear projections to match qubit count
        self.forget_lin = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, self.n_qubits)

    def forward(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(x, states)
        # Unroll over sequence length
        outputs = []
        for step in range(x.shape[1]):
            inp = x[:, step, :]
            combined = torch.cat([inp, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        stacked = torch.cat(outputs, dim=1)
        return stacked, (hx, cx)

    def _init_states(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = x.shape[0]
        device = x.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


__all__ = ["QuantumAttention", "QuantumLSTMCell"]
