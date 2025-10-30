"""Quantum‑enhanced LSTM with a simple circuit‑cutting strategy.

The module keeps the original API but adds a `cut` flag that splits the
quantum circuit into two subcircuits, each operating on a subset of the
qubits.  The outputs of the subcircuits are mapped to the hidden
dimension separately and then summed.  This demonstrates how a large
quantum circuit can be partitioned for execution on hardware with
limited qubit count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

try:
    from torchcrf import CRF  # optional dependency
except ImportError:  # pragma: no cover
    CRF = None  # type: ignore[assignment]

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell with optional circuit cutting.

    Parameters
    ----------
    input_dim : int
        Size of the input vector.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits used for the quantum component.
    alpha : float, default=1.0
        Mixing ratio between classical and quantum outputs.
    cut : bool, default=False
        If True, the quantum circuit is split into two subcircuits.
    """
    class QLayer(tq.QuantumModule):
        """Small quantum circuit that acts on a set of wires."""
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Simple ansatz: RX rotations on each wire followed by a chain of CNOTs
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # chain of CNOTs
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        alpha: float = 1.0,
        cut: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.alpha = float(alpha)
        self.cut = cut

        # Define quantum layers
        if cut and n_qubits >= 2:
            half = n_qubits // 2
            self.forget_q1 = self.QLayer(half)
            self.forget_q2 = self.QLayer(n_qubits - half)
            self.input_q1 = self.QLayer(half)
            self.input_q2 = self.QLayer(n_qubits - half)
            self.update_q1 = self.QLayer(half)
            self.update_q2 = self.QLayer(n_qubits - half)
            self.output_q1 = self.QLayer(half)
            self.output_q2 = self.QLayer(n_qubits - half)
            self.quantum_to_hidden1 = nn.Linear(half, hidden_dim)
            self.quantum_to_hidden2 = nn.Linear(n_qubits - half, hidden_dim)
        else:
            self.forget_q = self.QLayer(n_qubits)
            self.input_q = self.QLayer(n_qubits)
            self.update_q = self.QLayer(n_qubits)
            self.output_q = self.QLayer(n_qubits)
            self.quantum_to_hidden = nn.Linear(n_qubits, hidden_dim)

        # Linear gates for classical part
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Mapping from combined input to qubit space
        self.combined_to_qubits = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gates
            f_cls = torch.sigmoid(self.forget_linear(combined))
            i_cls = torch.sigmoid(self.input_linear(combined))
            g_cls = torch.tanh(self.update_linear(combined))
            o_cls = torch.sigmoid(self.output_linear(combined))

            # Quantum gates
            q_input = self.combined_to_qubits(combined)
            if self.cut and self.n_qubits >= 2:
                f_q1 = self.forget_q1(q_input)
                f_q2 = self.forget_q2(q_input)
                f_q = self.quantum_to_hidden1(f_q1) + self.quantum_to_hidden2(f_q2)

                i_q1 = self.input_q1(q_input)
                i_q2 = self.input_q2(q_input)
                i_q = self.quantum_to_hidden1(i_q1) + self.quantum_to_hidden2(i_q2)

                g_q1 = self.update_q1(q_input)
                g_q2 = self.update_q2(q_input)
                g_q = self.quantum_to_hidden1(g_q1) + self.quantum_to_hidden2(g_q2)

                o_q1 = self.output_q1(q_input)
                o_q2 = self.output_q2(q_input)
                o_q = self.quantum_to_hidden1(o_q1) + self.quantum_to_hidden2(o_q2)
            else:
                f_q = self.forget_q(q_input)
                i_q = self.input_q(q_input)
                g_q = self.update_q(q_input)
                o_q = self.output_q(q_input)
                f_q = self.quantum_to_hidden(f_q)
                i_q = self.quantum_to_hidden(i_q)
                g_q = self.quantum_to_hidden(g_q)
                o_q = self.quantum_to_hidden(o_q)

            # Map quantum outputs to hidden dimension and apply activation
            f_q = torch.sigmoid(f_q)
            i_q = torch.sigmoid(i_q)
            g_q = torch.tanh(g_q)
            o_q = torch.sigmoid(o_q)

            # Mixing
            f = self.alpha * f_cls + (1 - self.alpha) * f_q
            i = self.alpha * i_cls + (1 - self.alpha) * i_q
            g = self.alpha * g_cls + (1 - self.alpha) * g_q
            o = self.alpha * o_cls + (1 - self.alpha) * o_q

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a pure classical
    `nn.LSTM` and the quantum‑enhanced `QLSTM`.  An optional CRF layer
    can be appended when the `torchcrf` package is available.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        alpha: float = 1.0,
        cut: bool = False,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, alpha, cut)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.use_crf = use_crf and CRF is not None
        if self.use_crf:
            self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(1)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        if self.use_crf:
            scores = tag_logits.permute(1, 0, 2)
            decoded = self.crf.decode(scores)
            return torch.tensor(decoded, device=sentence.device, dtype=torch.long)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
