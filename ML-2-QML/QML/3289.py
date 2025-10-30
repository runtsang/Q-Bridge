"""Quantum‑enhanced LSTM with Qiskit self‑attention.

The module mirrors the classical HybridQLSTM but replaces the linear
gates with small parameterised quantum circuits and adds a Qiskit
self‑attention block that produces attention weights from a quantum
statevector.  It is fully compatible with the original interface
while exposing genuinely quantum behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, Aer


class QuantumSelfAttention:
    """Self‑attention block implemented with a Qiskit circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("statevector_simulator")

    def _build_circuit(self, rotation_params, entangle_params):
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def run(self, rotation_params, entangle_params, inputs):
        # inputs: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = inputs.shape
        weights = torch.zeros(batch_size, seq_len, device=inputs.device)
        for b in range(batch_size):
            circuit = self._build_circuit(rotation_params, entangle_params)
            result = qiskit.execute(circuit, self.backend).result()
            state = result.get_statevector(circuit)
            exp = []
            for q in range(self.n_qubits):
                plus = 0.0
                minus = 0.0
                for idx, amp in enumerate(state):
                    if ((idx >> q) & 1) == 0:
                        plus += abs(amp) ** 2
                    else:
                        minus += abs(amp) ** 2
                exp.append(plus - minus)
            exp = torch.tensor(exp, device=inputs.device, dtype=torch.float32)
            # truncate or expand to seq_len
            if seq_len <= self.n_qubits:
                exp = exp[:seq_len]
            else:
                exp = exp.expand(seq_len)
            weights[b] = exp
        weights = torch.softmax(weights, dim=1)
        weighted = torch.sum(weights.unsqueeze(-1) * inputs, dim=1)
        return weighted


class HybridQLSTM(nn.Module):
    """Quantum LSTM cell with an optional Qiskit self‑attention pre‑step."""
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

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_attention = use_attention

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        if self.use_attention:
            self.attention = QuantumSelfAttention(n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)

        # Apply quantum attention if enabled
        if self.use_attention:
            # Random parameters for demonstration
            rot = torch.rand(self.n_qubits * 3, device=inputs.device)
            ent = torch.rand(self.n_qubits - 1, device=inputs.device)
            inputs = self.attention.run(rot.cpu().numpy(), ent.cpu().numpy(), inputs)

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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`HybridQLSTM`."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            use_attention=use_attention,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
