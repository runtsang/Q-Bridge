"""Hybrid quantum LSTM module for sequence tagging and regression.

The implementation mirrors the classical counterpart but replaces each gate
with a small variational circuit that uses a random layer followed by
trainable RX/RY rotations.  The input embeddings are encoded into qubit
states via a GeneralEncoder.  The final hidden state is passed through a
linear head to produce either tag logits or a regression value.

This class is intentionally API‑compatible with the classical
HybridQLSTM – the same arguments are accepted and the same forward
signature is used.  The quantum backend is fully optional: if
``n_qubits`` is set to 0 the constructor will raise an error, encouraging
the user to use the classical implementation instead.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQLSTM(tq.QuantumModule):
    """
    Quantum LSTM with a regression or tagging head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embedding.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits (wires) used for the variational gates.
    task : str, {'tagging','regression'}, default 'tagging'
        Mode of operation.
    vocab_size : int, optional
        Vocabulary size for tagging.
    tagset_size : int, optional
        Number of tag classes for tagging.
    """

    class QGate(tq.QuantumModule):
        """
        Variational gate that outputs a vector of size ``n_qubits``.
        The circuit consists of a random layer followed by trainable
        RX and RY rotations on each wire, and a final layer of CNOTs
        that entangles the qubits.
        """

        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Random layer introduces expressivity
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            # Entangle wires in a chain and close the loop
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        task: str = "tagging",
        vocab_size: Optional[int] = None,
        tagset_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive for the quantum backend.")
        self.task = task
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to feed quantum gates
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Encoder to map classical embeddings into qubit amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_qubits}xRy"])

        # Heads
        if task == "tagging":
            if vocab_size is None or tagset_size is None:
                raise ValueError("vocab_size and tagset_size required for tagging")
            self.word_embeddings = nn.Embedding(vocab_size, input_dim)
            self.head = nn.Linear(hidden_dim, tagset_size)
        elif task == "regression":
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unsupported task {task!r}")

    def _apply_gate(
        self,
        raw: torch.Tensor,
        gate_mod: tq.QuantumModule,
    ) -> torch.Tensor:
        """
        Apply a quantum gate to the given device and return a probability
        vector of shape (batch_size,).
        """
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=raw.shape[0],
            device=raw.device,
        )
        self.encoder(qdev, raw)
        gate_mod(qdev)
        probs = tqf.expectation(
            qdev,
            ops=[tq.PauliZ] * self.n_qubits,
            wires=list(range(self.n_qubits)),
        )
        # Collapse multi‑qubit expectation to a scalar per batch element
        return probs.mean(dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            * Tagging: LongTensor of shape [seq_len, batch_size] or
              FloatTensor of shape [seq_len, batch_size, input_dim].
            * Regression: FloatTensor of shape [seq_len, batch_size, input_dim].
        states : tuple of torch.Tensor, optional
            Initial hidden and cell states (h_0, c_0).

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            * Tagging: logits of shape [seq_len, batch_size, tagset_size]
            * Regression: predictions of shape [batch_size]
        """
        if self.task == "tagging":
            # Embed token indices
            embeds = self.word_embeddings(inputs)
            seq_len, batch_size, _ = embeds.shape
        else:
            embeds = inputs
            seq_len, batch_size, _ = embeds.shape

        h, c = self._init_states(batch_size, embeds.device)
        if states is not None:
            h, c = states

        outputs = []
        for t in range(seq_len):
            x_t = embeds[t]
            combined = torch.cat([x_t, h], dim=1)  # shape: [batch, hidden_dim + input_dim]
            f_raw = self.linear_forget(combined)
            i_raw = self.linear_input(combined)
            g_raw = self.linear_update(combined)
            o_raw = self.linear_output(combined)

            f = torch.sigmoid(self._apply_gate(f_raw, self.forget_gate))
            i = torch.sigmoid(self._apply_gate(i_raw, self.input_gate))
            g = torch.tanh(self._apply_gate(g_raw, self.update_gate))
            o = torch.sigmoid(self._apply_gate(o_raw, self.output_gate))

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # shape: [seq_len, batch, hidden_dim]

        if self.task == "tagging":
            logits = self.head(outputs)
            return logits, (h, c)
        else:  # regression
            pred = self.head(h).squeeze(-1)
            return pred, (h, c)

    def _init_states(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h0, c0

__all__ = ["HybridQLSTM"]
