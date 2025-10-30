from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HybridQLSTM(nn.Module):
    """
    A hybrid LSTM cell that blends classical linear gates with quantum‑enhanced gates.
    The blending is controlled by a scalar ``quantum_strength`` that can be fixed or learned.
    The cell retains the original QLSTM interface for drop‑in compatibility.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        *,
        quantum_strength: float = 0.0,
        pretrain_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.quantum_strength = nn.Parameter(
            torch.tensor([quantum_strength], dtype=torch.float32)
        ) if pretrain_quantum else torch.tensor([quantum_strength], dtype=torch.float32)

        # Classical linear layers for each gate
        self._lin_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self._lin_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self._lin_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self._lin_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate placeholders (simple QML wrappers)
        self._q_forget = self._create_quantum_gate()
        self._q_input = self._create_quantum_gate()
        self._q_update = self._create_quantum_gate()
        self._q_output = self._create_quantum_gate()

    def _create_quantum_gate(self) -> nn.Module:
        # Simple parameterized quantum layer using a single qubit
        # for demonstration; replace with a full circuit if needed.
        return nn.Sequential(
            nn.Linear(self.n_qubits, self.n_qubits),
            nn.Tanh()
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

            # Classical gate outputs
            f_c = torch.sigmoid(self._lin_forget(combined))
            i_c = torch.sigmoid(self._lin_input(combined))
            g_c = torch.tanh(self._lin_update(combined))
            o_c = torch.sigmoid(self._lin_output(combined))

            # Quantum gate outputs (scaled to [0,1] or [-1,1] as needed)
            f_q = torch.sigmoid(self._q_forget(combined))
            i_q = torch.sigmoid(self._q_input(combined))
            g_q = torch.tanh(self._q_update(combined))
            o_q = torch.sigmoid(self._q_output(combined))

            # Blend classical and quantum contributions
            f = (1 - self.quantum_strength) * f_c + self.quantum_strength * f_q
            i = (1 - self.quantum_strength) * i_c + self.quantum_strength * i_q
            g = (1 - self.quantum_strength) * g_c + self.quantum_strength * g_q
            o = (1 - self.quantum_strength) * o_c + self.quantum_strength * o_q

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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

class HybridLSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical, quantum, or hybrid LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        *,
        quantum_strength: float = 0.0,
        pretrain_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits,
                quantum_strength=quantum_strength,
                pretrain_quantum=pretrain_quantum,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
