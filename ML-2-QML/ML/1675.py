"""Hybrid classical‑quantum LSTM implementation.

This module extends the original seed by adding:
- A configurable `mode` ('classical', 'quantum', 'hybrid') and `mix_factor` blending quantum and classical gates.
- A lightweight quantum sub‑module that can be toggled on demand.
- A consistent interface compatible with the original LSTMTagger.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:
    # torchquantum is optional; quantum mode will not be available
    tq = None
    tqf = None


class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell that can operate in classical, quantum, or hybrid mode."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        mode: str = "hybrid",
        mix_factor: float = 0.5,
    ) -> None:
        super().__init__()
        assert mode in ("classical", "quantum", "hybrid"), "mode must be one of 'classical', 'quantum', 'hybrid'"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.mode = mode
        self.mix_factor = mix_factor

        # Classical linear layers for gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if mode in ("quantum", "hybrid"):
            # Ensure compatibility
            assert hidden_dim == n_qubits, "For quantum/hybrid mode, hidden_dim must equal n_qubits"
            # Quantum layers
            self.forget_q = self._build_qgate()
            self.input_q = self._build_qgate()
            self.update_q = self._build_qgate()
            self.output_q = self._build_qgate()

    def _build_qgate(self) -> nn.Module:
        """Build a simple variational quantum gate."""
        if tq is None:
            raise RuntimeError("torchquantum is required for quantum modes")
        class QGate(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                # Parameterized rotation gates
                self.params = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                # Encode each qubit with an RX gate using input value
                for wire in range(min(self.n_wires, x.shape[1])):
                    tq.RX(x[:, wire])(qdev, wires=wire)
                # Apply trainable gates
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                # Entangle wires
                for wire in range(self.n_wires - 1):
                    tqf.CNOT(qdev, wires=[wire, wire + 1])
                return self.measure(qdev)
        return QGate(self.n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.mode in ("quantum", "hybrid"):
                # Quantum gate outputs
                q_f = torch.sigmoid(self.forget_q(self.forget_linear(combined)))
                q_i = torch.sigmoid(self.input_q(self.input_linear(combined)))
                q_g = torch.tanh(self.update_q(self.update_linear(combined)))
                q_o = torch.sigmoid(self.output_q(self.output_linear(combined)))

                if self.mode == "hybrid":
                    # Blend with classical gates
                    f = self.mix_factor * q_f + (1 - self.mix_factor) * f
                    i = self.mix_factor * q_i + (1 - self.mix_factor) * i
                    g = self.mix_factor * q_g + (1 - self.mix_factor) * g
                    o = self.mix_factor * q_o + (1 - self.mix_factor) * o
                else:  # quantum mode
                    f, i, g, o = q_f, q_i, q_g, q_o

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the hybrid LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        mode: str = "hybrid",
        mix_factor: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                mode=mode,
                mix_factor=mix_factor,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
