"""Quantum‑enhanced LSTM layers for sequence tagging.

This module implements the same interface as the classical
``QLSTM__gen361`` but replaces the optional quantum‑style gate
with a true variational quantum circuit implemented via
``torchquantum``.  The circuit is small, entangling a chain of
qubits and returning a measurement vector that can be fused
into the LSTM cell.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """
    Small variational circuit that encodes an input vector and
    returns a measurement vector.  The circuit consists of:
      * RX rotation per input element (encoding)
      * Trainable RX gates (parameter layer)
      * A CNOT chain that entangles all wires
      * Measurement of all wires in the Z basis
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each input element as an RX rotation on its wire
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        # Trainable RX layers
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        # Entangling CNOT chain (last wire to first to close the loop)
        self.cnot_chain = nn.ModuleList()
        for i in range(n_wires - 1):
            self.cnot_chain.append(tq.CNOT(wires=[i, i + 1]))
        self.cnot_chain.append(tq.CNOT(wires=[n_wires - 1, 0]))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, n_wires).

        Returns
        -------
        torch.Tensor
            Measurement vector of shape (batch, n_wires).
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        for cnot in self.cnot_chain:
            cnot(qdev)
        return self.measure(qdev)


class QLSTM__gen361(nn.Module):
    """
    Quantum‑enhanced LSTM cell that replaces the optional
    classical quantum‑style gate with a variational quantum circuit.
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

        # Gate layers: either a variational quantum circuit or identity
        if self.use_quantum_gate and self.n_qubits > 0:
            self.forget_gate = QLayer(self.n_qubits)
            self.input_gate = QLayer(self.n_qubits)
            self.update_gate = QLayer(self.n_qubits)
            self.output_gate = QLayer(self.n_qubits)
        else:
            # Identity modules that simply pass through the input
            self.forget_gate = nn.Identity()
            self.input_gate = nn.Identity()
            self.update_gate = nn.Identity()
            self.output_gate = nn.Identity()

        # Linear projections to the quantum circuit input space
        self.forget_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, self.n_qubits)

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

            # Classical linear projections
            f_lin = self.forget_linear(combined)
            i_lin = self.input_linear(combined)
            g_lin = self.update_linear(combined)
            o_lin = self.output_linear(combined)

            # Quantum gate (or identity)
            f = torch.sigmoid(self.forget_gate(f_lin))
            i = torch.sigmoid(self.input_gate(i_lin))
            g = torch.tanh(self.update_gate(g_lin))
            o = torch.sigmoid(self.output_gate(o_lin))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


class LSTMTagger__gen361(nn.Module):
    """
    Sequence tagging model that can switch between a standard
    LSTM and the quantum QLSTM__gen361.
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
