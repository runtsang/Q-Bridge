"""Enhanced classical LSTM with optional quantum gate injection.

This module defines a hybrid LSTM that can optionally use a
parameterised “quantum” gate for each cell gate.  The gate is
implemented as a linear layer that returns a vector of size
`hidden_dim`.  The gate can be swapped out for a true quantum
module in the `qml` implementation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM__gen361(nn.Module):
    """
    Classical LSTM cell that optionally injects a quantum‑parameterised
    gate into each gate’s linear pre‑activation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int, default 0
        Number of qubits that would be used by the quantum gate.
        If ``n_qubits`` is zero, the cell behaves like a standard
        classical LSTM.
    use_quantum_gate : bool, default False
        When ``True`` and ``n_qubits`` > 0, a simple
        ``nn.Linear`` is used to emulate a quantum gate.  This
        emulation is purely classical but preserves the interface
        of the quantum version.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_quantum_gate: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum_gate = use_quantum_gate

        # Linear layers for the four gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional quantum‑style gate
        if self.use_quantum_gate and self.n_qubits > 0:
            # This linear layer emulates a quantum gate that maps
            # a vector of size `n_qubits` to the hidden dimension.
            self.quantum_gate = nn.Linear(self.n_qubits, hidden_dim)
        else:
            self.quantum_gate = None

    def _apply_quantum_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the emulated quantum gate to a tensor of shape
        (batch, n_qubits).  The result is a vector of shape
        (batch, hidden_dim).
        """
        if self.quantum_gate is None:
            raise RuntimeError("Quantum gate not configured.")
        return self.quantum_gate(x)

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
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).
        states : tuple of torch.Tensor, optional
            Initial hidden and cell states.  If None, zeros are used.

        Returns
        -------
        outputs : torch.Tensor
            Hidden states for each time step (seq_len, batch, hidden_dim).
        final_state : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            # Inject the quantum gate if configured
            if self.quantum_gate is not None:
                # Use a simple projection of the combined vector to `n_qubits`
                q_input = combined[:, : self.n_qubits]
                q_gate_out = self._apply_quantum_gate(q_input)
                # Element‑wise blend the classical gate with the quantum output
                f = f * q_gate_out
                i = i * q_gate_out
                g = g * q_gate_out
                o = o * q_gate_out

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


class LSTMTagger__gen361(nn.Module):
    """
    Sequence tagging model that can switch between a standard
    LSTM and the hybrid QLSTM__gen361.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum_gate: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0 and use_quantum_gate:
            self.lstm = QLSTM__gen361(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                use_quantum_gate=True,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices with shape (seq_len, batch).

        Returns
        -------
        log_probs : torch.Tensor
            Log‑softmax of tag scores (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["QLSTM__gen361", "LSTMTagger__gen361"]
