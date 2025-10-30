"""Quantum‑enhanced hybrid LSTM with per‑gate quantum circuits.

HybridQLSTM extends the classical version by allowing any of the four LSTM
gates to be realised by a small variational quantum circuit.  The class
provides a drop‑in interface for sequence tagging and other recurrent
tasks and is fully compatible with PyTorch's autograd.

Key quantum features
--------------------
- **Gate‑specific quantum mode** – each gate can be toggled to a small
  variational circuit (RX rotations on each qubit followed by a
  CNOT‑entangling layer).
- **Batch‑wise execution** – the circuit is executed on a batched
  :class:`torchquantum.QuantumDevice`.
- **Dropout** – optional per‑gate dropout before the quantum or linear
  transformation.
- **Orthogonal initialization** – all trainable parameters are
  initialized with orthogonal matrices or uniform distributions.

The implementation is self‑contained and can be imported as a normal
Python module.

"""

from __future__ import annotations

from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Minimal variational circuit that maps an input vector onto qubits.

    The circuit encodes each input element ``x_i`` as an ``RX(x_i)`` on wire
    ``i``, applies a trainable ``RX`` gate on each wire, entangles all wires
    with a linear CNOT chain, and finally measures all qubits in the Pauli‑Z
    basis.
    """

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires

        # Encode input values as rotation angles
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )

        # Trainable RX gates (one per wire)
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a batch of measurement outcomes."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Encode input
        self.encoder(qdev, x)
        # Trainable rotations
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entanglement: linear CNOT chain
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell that can switch any gate to a quantum circuit.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int, default 0
        Number of qubits per quantum gate.  Must equal ``hidden_dim`` when
        any gate is quantum.
    gate_mode : dict[str, bool], optional
        Mapping ``{'forget': bool, 'input': bool, 'update': bool, 'output': bool}``
        specifying whether each gate should be quantum (`True`) or classical
        (`False`).  Defaults to all ``False``.
    dropout : float, optional
        Dropout probability applied to each gate's linear output before
        passing it to a quantum circuit or activation.  ``0`` disables.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        gate_mode: Optional[Dict[str, bool]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = dropout

        if gate_mode is None:
            gate_mode = {
                "forget": False,
                "input": False,
                "update": False,
                "output": False,
            }
        self.gate_mode = gate_mode

        # Linear projections for all gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)

        # Initialise linear layers orthogonally
        for lin in (
            self.forget_linear,
            self.input_linear,
            self.update_linear,
            self.output_linear,
        ):
            nn.init.orthogonal_(lin.weight)
            nn.init.zeros_(lin.bias)

        # Quantum modules for gates requested as quantum
        self.qgate = {}
        if n_qubits > 0:
            for gate_name, use_q in gate_mode.items():
                if use_q:
                    # Guard against hidden_dim mismatch
                    if n_qubits!= hidden_dim:
                        raise ValueError(
                            f"n_qubits ({n_qubits}) must equal hidden_dim ({hidden_dim}) "
                            "when a gate is quantum."
                        )
                    self.qgate[gate_name] = QLayer(n_qubits)

    def _dropout(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dropout > 0.0:
            return F.dropout(tensor, p=self.dropout, training=self.training)
        return tensor

    def _init_states(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len, batch_size, _ = inputs.shape
        device = inputs.device

        hx, cx = self._init_states(batch_size, device) if states is None else states

        outputs = []

        for t in range(seq_len):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)

            # Forget gate
            f_lin = self._dropout(self.forget_linear(combined))
            if self.gate_mode.get("forget", False):
                f_lin = self.qgate["forget"](f_lin)
            f = torch.sigmoid(f_lin)

            # Input gate
            i_lin = self._dropout(self.input_linear(combined))
            if self.gate_mode.get("input", False):
                i_lin = self.qgate["input"](i_lin)
            i = torch.sigmoid(i_lin)

            # Update gate
            g_lin = self._dropout(self.update_linear(combined))
            if self.gate_mode.get("update", False):
                g_lin = self.qgate["update"](g_lin)
            g = torch.tanh(g_lin)

            # Output gate
            o_lin = self._dropout(self.output_linear(combined))
            if self.gate_mode.get("output", False):
                o_lin = self.qgate["output"](o_lin)
            o = torch.sigmoid(o_lin)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`HybridQLSTM`."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        gate_mode: Optional[Dict[str, bool]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            gate_mode=gate_mode,
            dropout=dropout,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["HybridQLSTM", "LSTMTagger"]
