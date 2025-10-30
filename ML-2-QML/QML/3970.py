"""Hybrid QLSTM implementation using torchquantum.

The quantum version closely follows the original QLSTM.py but is
re‑structured to expose a clean, drop‑in replacement that can be
imported as ``HybridQLSTM``.  The key differences are:

* A dedicated ``QLayer`` that wraps a small parametric circuit
  (GeneralEncoder + RX + CNOT) and returns the Pauli‑Z expectation
  for each qubit.
* All gates (forget, input, update, output) are instantiated
  with this QLayer.
* The class is fully differentiable via the torchquantum
  autograd support and can be trained on a GPU using the
  `torchquantum` backend.

Usage
-----
>>> from qml_code import HybridQLSTM
>>> model = HybridQLSTM(input_dim=128, hidden_dim=256, n_qubits=4)
>>> out, (h,c) = model(inputs)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# QLayer – quantum circuit for a single gate
# --------------------------------------------------------------------------- #
class QLayer(nn.Module):
    """Quantum circuit that implements a parametric gate.

    The circuit encodes *n_wires* input parameters into the qubits
    via RX rotations, applies a trainable RX on each wire, and
    entangles the qubits with a simple CNOT chain.  The Pauli‑Z
    expectation value of each qubit is returned as a vector.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires

        # Encoder that maps classical input to quantum rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        # Trainable rotations
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        # Measurement of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is expected to be of shape (batch, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entangle the wires with a CNOT chain
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
# HybridQLSTM – quantum LSTM with classical mapping to hidden dimension
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """Drop‑in replacement for the original QLSTM.

    The module builds a standard LSTM where each gate is implemented
    by a *QLayer* followed by a linear mapping to the hidden dimension.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Gate‑specific QLayers
        self.forget = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear transforms from concatenated input/state to gate parameters
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Mapping from gate output (n_qubits) to hidden dimension
        self.map_forget = nn.Linear(n_qubits, hidden_dim)
        self.map_input = nn.Linear(n_qubits, hidden_dim)
        self.map_update = nn.Linear(n_qubits, hidden_dim)
        self.map_output = nn.Linear(n_qubits, hidden_dim)

    # --------------------------------------------------------------------- #
    # Helper to initialise hidden/cell states
    # --------------------------------------------------------------------- #
    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_raw = self.forget(self.linear_forget(combined))
            f = torch.sigmoid(self.map_forget(f_raw))

            i_raw = self.input_gate(self.linear_input(combined))
            i = torch.sigmoid(self.map_input(i_raw))

            g_raw = self.update(self.linear_update(combined))
            g = torch.tanh(self.map_update(g_raw))

            o_raw = self.output(self.linear_output(combined))
            o = torch.sigmoid(self.map_output(o_raw))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)

# --------------------------------------------------------------------------- #
# LSTMTagger – sequence tagging model using HybridQLSTM
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
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

        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
