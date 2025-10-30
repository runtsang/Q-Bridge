"""Hybrid LSTM with quantum gates and a Qiskit‑based self‑attention block.

Key features:
* Quantum logic for the four LSTM gates is implemented via torchquantum.
* A dedicated `QuantumSelfAttention` class builds a small Qiskit circuit
  that acts as a self‑attention mechanism; the rotation angles are extracted
  from the hidden state.
* When `n_qubits=0` the class falls back to classical linear layers,
  keeping the API identical to the seed implementation.
* Attention weights are derived from the measurement statistics of the
  quantum circuit and used to re‑weight the hidden sequence.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute


class QuantumSelfAttention:
    """Qiskit implementation of a self‑attention block.

    The circuit applies RX/RY/RZ rotations determined by the rotation_params
    and a simple CNOT entanglement pattern controlled by entangle_params.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def _counts_to_weight(self, counts: Dict[str, int]) -> float:
        """Return a scalar weight based on measurement statistics."""
        shots = sum(counts.values())
        # Weight as proportion of bitstrings with the first qubit = 1
        weight = sum(c for b, c in counts.items() if b[-1] == "1") / shots
        return weight

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 512,
    ) -> float:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        return self._counts_to_weight(counts)


class QLSTM(nn.Module):
    """Hybrid LSTM cell with quantum gates and optional quantum self‑attention."""

    class QLayer(tq.QuantumModule):
        """Small quantum gate used inside each LSTM gate."""

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
        n_qubits: int = 0,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_attention = use_attention

        if n_qubits > 0:
            self.forget = self.QLayer(n_qubits)
            self.input = self.QLayer(n_qubits)
            self.update = self.QLayer(n_qubits)
            self.output = self.QLayer(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if use_attention:
            self.attention = QuantumSelfAttention(n_qubits=4)
            self.attn_rot_params = nn.Parameter(torch.randn(12))
            self.attn_ent_params = nn.Parameter(torch.randn(3))

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input(combined))
                g = torch.tanh(self.update(combined))
                o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        if self.use_attention:
            attn_weights = []
            for hx_t in outputs.unbind(dim=0):
                rot = hx_t[:12].detach().cpu().numpy()
                ent = hx_t[12:15].detach().cpu().numpy()
                weight = self.attention.run(rot, ent, shots=256)
                attn_weights.append(weight)
            attn_weights = torch.tensor(attn_weights, device=outputs.device)
            attn_weights = F.softmax(attn_weights, dim=0)
            context = torch.sum(outputs * attn_weights.unsqueeze(1), dim=0, keepdim=True)
            outputs = outputs + context

        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """Tagger that can switch between classical, quantum, and attention modes."""

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
        self.lstm = QLSTM(
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


__all__ = ["QLSTM", "LSTMTagger"]
