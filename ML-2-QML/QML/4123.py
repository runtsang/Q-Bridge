"""Quantum LSTM tagger with quantum convolution and attention.

This module extends the seed quantum LSTM by adding:
  • A Qiskit 2×2 convolution filter that converts classical data into
    measurement‑based scalars.
  • A Qiskit self‑attention circuit that produces a probabilistic
    representation of the input sequence.
  • A projection layer that maps the concatenated classical + quantum
    features into the hidden dimension of the quantum LSTM.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

# ------------------------------------------------------------------
#  Quantum convolution (quanvolution) filter
# ------------------------------------------------------------------
class QuantumConv:
    """
    A 2×2 quantum filter that maps a classical patch to a scalar
    probability of measuring |1> across the qubits.
    """
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(theta):
            qc.rx(t, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Run the filter on a 2×2 array of classical values.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {p: np.pi if v > self.threshold else 0 for p, v in zip(self.circuit.parameters, dat)}
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        return counts / (self.shots * self.n_qubits)

# ------------------------------------------------------------------
#  Quantum self‑attention circuit
# ------------------------------------------------------------------
class QuantumAttention:
    """
    Qiskit circuit that implements a simple self‑attention style block.
    """
    def __init__(self, n_qubits: int = 4, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

# ------------------------------------------------------------------
#  Quantum LSTM cell (variational gates)
# ------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Quantum LSTM cell where each gate is implemented by a small
    variational circuit on n_qubits qubits.
    """
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
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM gate
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections to the quantum register
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

        return torch.cat(outputs, dim=0), (hx, cx)

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

# ------------------------------------------------------------------
#  Tagger that chains quantum conv → attention → LSTM
# ------------------------------------------------------------------
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that processes data through a quantum
    convolution filter, a quantum attention circuit, and then a
    quantum LSTM cell.  The architecture mirrors the classical
    :class:`LSTMTagger` but replaces each preprocessing block with
    its quantum counterpart.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Quantum preprocessing blocks
        self.conv = QuantumConv()
        self.attn = QuantumAttention()

        # LSTM layer
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: Tensor of word indices, shape (seq_len, batch)
        Returns:
            log‑probabilities over tags for each token
        """
        # Word embeddings
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)

        # Quantum convolution filter
        conv_feats = torch.tensor(
            [self.conv.run(e.cpu().numpy().reshape(2, 2)) for e in embeds],
            dtype=torch.float32,
        ).unsqueeze(-1)  # (seq_len, batch, 1)

        # Quantum attention circuit
        rotation = np.random.rand(12)          # 4 qubits × 3 rotation params
        entangle = np.random.rand(3)           # 3 entanglement params
        attn_counts = self.attn.run(rotation, entangle)
        # Convert counts to a simple float feature
        attn_float = sum(sum(int(bit) for bit in k) * v for k, v in attn_counts.items())
        attn_tensor = torch.tensor(attn_float, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # Concatenate all features
        lstm_input = torch.cat([embeds, conv_feats, attn_tensor], dim=-1)

        # LSTM (classical or quantum)
        lstm_out, _ = self.lstm(lstm_input.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
