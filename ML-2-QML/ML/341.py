"""Hybrid classical‑quantum LSTM with depth‑controlled variational gates.

This module keeps the original public API (`QLSTM` and `LSTMTagger`) but adds a
quantum‑aware encoder that can be toggled on/off and whose depth can be tuned.
The implementation is deliberately lightweight: each gate is a small
variational circuit that runs on a CPU‑based simulator.  The module can be used
in a pure‑Python training loop or plugged into a larger PyTorch pipeline.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum backend
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception:  # pragma: no cover
    tq = None  # type: ignore
    tqf = None  # type: ignore


class _VarGate(nn.Module):
    """Variational quantum gate that maps a classical vector to a quantum circuit.

    The circuit depth is controlled by ``depth``.  The output is the expectation
    value of Pauli‑Z on each wire, which is used as the gate activation.
    """

    def __init__(
        self,
        in_dim: int,
        n_wires: int,
        depth: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if tq is None:
            raise RuntimeError("torchquantum is required for _VarGate")
        self.in_dim = in_dim
        self.n_wires = n_wires
        self.depth = depth
        self.device = device or torch.device("cpu")

        # Linear map from classical input to qubit angles
        self.lin = nn.Linear(in_dim, n_wires, bias=False)

        # Parameters for each RX gate per depth layer
        self.params = nn.Parameter(
            torch.randn(depth, n_wires, 1, device=self.device)
        )

        # Store last expectation values for monitoring
        self.last_expect = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return expectation values of Pauli‑Z for each wire."""
        batch_size = x.size(0)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=batch_size, device=self.device
        )

        # Encode classical data as RX rotations
        angles = self.lin(x)  # (batch, n_wires)
        for wire in range(self.n_wires):
            tq.RX(angles[:, wire], wires=wire)(qdev)

        # Variational layers
        for d in range(self.depth):
            for wire in range(self.n_wires):
                tq.RX(self.params[d, wire, 0], wires=wire)(qdev)
            # Simple chain of CNOTs
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])

        # Measure all qubits in Z basis
        meas = tq.MeasureAll(tq.PauliZ)(qdev)
        self.last_expect = meas
        return meas


class QLSTM(nn.Module):
    """LSTM cell where each gate is realised by a small variational quantum circuit.

    If ``n_qubits`` is zero or torchquantum is unavailable, the module falls back
    to a purely classical implementation that mimics the original QLSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        if n_qubits > 0 and tq is not None:
            if hidden_dim!= n_qubits:
                raise ValueError("In quantum mode, hidden_dim must equal n_qubits")
            # quantum gates
            self.forget = _VarGate(input_dim + hidden_dim, n_qubits, depth)
            self.input = _VarGate(input_dim + hidden_dim, n_qubits, depth)
            self.update = _VarGate(input_dim + hidden_dim, n_qubits, depth)
            self.output = _VarGate(input_dim + hidden_dim, n_qubits, depth)

            # linear maps to qubit space
            self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            # classical fallback
            self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Standard LSTM forward pass with optional quantum gates."""
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            if self.n_qubits > 0 and tq is not None:
                f = torch.sigmoid(self.forget(self.lin_forget(combined)))
                i = torch.sigmoid(self.input(self.lin_input(combined)))
                g = torch.tanh(self.update(self.lin_update(combined)))
                o = torch.sigmoid(self.output(self.lin_output(combined)))
            else:
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input(combined))
                g = torch.tanh(self.update(combined))
                o = torch.sigmoid(self.output(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def get_expectations(self) -> dict[str, torch.Tensor]:
        """Return the last expectation values of each quantum gate.
        Only available when the module is in quantum mode.
        """
        if self.n_qubits == 0 or tq is None:
            raise RuntimeError("No quantum gates present")
        return {
            "forget": self.forget.last_expect,
            "input": self.input.last_expect,
            "update": self.update.last_expect,
            "output": self.output.last_expect,
        }


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
