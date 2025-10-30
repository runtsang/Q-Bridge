"""Hybrid LSTM with a quantum sampler for the forget gate, built on torchquantum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QSampler(tq.QuantumModule):
    """Quantum module that maps a 2‑dim input to a 2‑dim probability vector."""
    def __init__(self, n_wires: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode two classical inputs via RX rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
            ]
        )
        # Trainable RX gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        # Measure all wires in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (..., 2)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Expectation values of PauliZ
        expz = self.measure(qdev)  # shape (batch, n_wires)
        # Convert to probabilities in [0,1] and apply softmax
        probs = (expz + 1) / 2
        probs = F.softmax(probs, dim=-1)
        return probs


class HybridQLSTM(nn.Module):
    """Quantum‑augmented LSTM that uses a quantum sampler for the forget gate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, 2)
        self.input_linear   = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear  = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum sampler for forget gate
        self.forget_sampler = QSampler(n_wires=2)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_logits = self.forget_linear(combined)          # (..., 2)
            f_probs  = self.forget_sampler(f_logits)          # (..., 2)
            f = f_probs[:, 0]                                # use first probability
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
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


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the hybrid quantum‑classical LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim) if use_quantum else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
