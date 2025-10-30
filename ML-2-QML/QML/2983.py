"""Quantum Self‑Attention + LSTM hybrid module using Qiskit and TorchQuantum."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QuantumSelfAttention(nn.Module):
    """Quantum circuit that produces a self‑attention style output."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, 'q')
        self.cr = ClassicalRegister(n_qubits, 'c')
        self.backend = Aer.get_backend('qasm_simulator')

    def _build_circuit(self, rot_params: np.ndarray, ent_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rot_params[3*i], self.qr[i])
            qc.ry(rot_params[3*i+1], self.qr[i])
            qc.rz(rot_params[3*i+2], self.qr[i])
        for i in range(self.n_qubits-1):
            qc.crx(ent_params[i], self.qr[i], self.qr[i+1])
        qc.measure(self.qr, self.cr)
        return qc

    def forward(self, inputs: torch.Tensor, rot_params: np.ndarray, ent_params: np.ndarray,
                shots: int = 1024) -> torch.Tensor:
        # For simplicity: ignore inputs and return probability distribution from circuit
        qc = self._build_circuit(rot_params, ent_params)
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts(qc)
        probs = torch.tensor([counts.get(f'{i:0{self.n_qubits}b}', 0) for i in range(2**self.n_qubits)],
                             dtype=torch.float32)
        probs /= probs.sum()
        return probs.unsqueeze(0)  # (1, 2^n_qubits)

class QuantumQLSTM(nn.Module):
    """LSTM cell where gates are realised by TorchQuantum variational circuits."""
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
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_wires):
                tgt = 0 if w == self.n_wires-1 else w+1
                tqf.cnot(qdev, wires=[w, tgt])
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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
        out = torch.cat(outputs, dim=0)
        return out, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))

class SelfAttentionQLSTM(nn.Module):
    """Hybrid quantum module: quantum self‑attention followed by a quantum LSTM."""
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, num_heads: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = QuantumSelfAttention(n_qubits)
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embed_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor, rot_params: np.ndarray,
                ent_params: np.ndarray, shots: int = 1024) -> torch.Tensor:
        embeds = self.embedding(sentence)            # (seq_len, batch, embed_dim)
        attn_out = self.attention(embeds, rot_params, ent_params, shots)
        seq_len = embeds.size(0)
        batch = embeds.size(1)
        lstm_input = attn_out.repeat(seq_len, 1, 1)  # (seq_len, batch, 2^n_qubits)
        lstm_out, _ = self.lstm(lstm_input)
        logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(logits, dim=1)

__all__ = ["SelfAttentionQLSTM", "QuantumSelfAttention", "QuantumQLSTM"]
