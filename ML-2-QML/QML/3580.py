"""Quantum‑enhanced LSTM layers for sequence tagging with stochastic sampling.

The quantum variant mirrors the classical implementation but replaces each
gate with a parameterised quantum sub‑circuit.  A SamplerQNN is used to
produce a probability that gates the quantum output, acting as a
stochastic regulariser.  The design keeps the public API identical to
the original QLSTM.py, enabling seamless switching between classical
and quantum back‑ends.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
# SamplerQNN – probability mask generator (identical to the classical one)
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """
    Outputs a single scalar probability that gates the quantum gate outputs.
    Input shape: (batch, input_dim + hidden_dim)
    Output shape: (batch, 1)
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(combined))  # (batch, 1)


# --------------------------------------------------------------------------- #
# Quantum gate sub‑module (QLayer)
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """
    Small quantum circuit that maps an n‑qubit state to a probability vector.
    The circuit consists of a trainable RX rotation on each qubit followed by
    a chain of CNOTs to entangle the wires, then a measurement in the Z basis.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder: encode each input dimension into a single RX gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Trainable rotations
        self.rxs = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        # Entanglement layer
        self.cnot_chain = [
            (i, (i + 1) % n_wires) for i in range(n_wires)
        ]
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.rxs):
            gate(qdev, wires=wire)
        for src, tgt in self.cnot_chain:
            tqf.cnot(qdev, wires=[src, tgt])
        return self.measure(qdev)  # (batch, n_wires)


# --------------------------------------------------------------------------- #
# Quantum LSTM with optional stochastic gating
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """
    LSTM cell where the four gates are realised by small quantum circuits.
    If ``use_sampler`` is True, a SamplerQNN scales the quantum outputs
    with a learned probability, providing a stochastic regulariser.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_sampler = use_sampler

        # Quantum gates
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Classical linear layers that feed the quantum circuits
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        if self.use_sampler:
            self.sampler = SamplerQNN(input_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classic linear pre‑processing
            f_in = self.linear_forget(combined)
            i_in = self.linear_input(combined)
            g_in = self.linear_update(combined)
            o_in = self.linear_output(combined)

            # Quantum processing
            f = torch.sigmoid(self.forget(f_in))
            i = torch.sigmoid(self.input(i_in))
            g = torch.tanh(self.update(g_in))
            o = torch.sigmoid(self.output(o_in))

            if self.use_sampler:
                p = self.sampler(combined)  # (batch, 1)
                f *= p
                i *= p
                g *= p
                o *= p

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
# Sequence tagging model
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    The ``use_sampler`` flag activates the stochastic gating in the quantum LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, use_sampler=use_sampler)
        else:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=0, use_sampler=use_sampler)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
