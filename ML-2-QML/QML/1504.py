"""Quantum‑enhanced LSTM layer using torchquantum with optional Pennylane or Qiskit support.

The class `QLSTM__` mirrors the classical API but replaces the linear gates by
parameterised quantum circuits.  Each gate is a `QLayer` that applies a linear
embedding of the concatenated input and hidden state into rotation angles for the circuit.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum back‑ends
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception:  # pragma: no cover
    tq = None  # type: ignore
    tqf = None  # type: ignore

try:
    import pennylane as qml
    import pennylane.numpy as pnp
except Exception:  # pragma: no cover
    qml = None  # type: ignore
    pnp = None  # type: ignore

try:
    import qiskit
except Exception:  # pragma: no cover
    qiskit = None  # type: ignore


class QLayer(nn.Module):
    """
    Variational quantum circuit that can be executed on multiple back‑ends.
    Parameters
    ----------
    n_wires : int
        Number of qubits used by the circuit.
    backend : str, optional
        Quantum back‑end.  Options are ``"torchquantum"``, ``"pennylane"``, or ``"qiskit"``.
        Defaults to ``"torchquantum"``.
    """

    def __init__(self, n_wires: int, backend: str = "torchquantum") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.backend = backend

        if backend == "torchquantum":
            # Linear encoder that maps the input vector to rotation angles
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)
        elif backend == "pennylane":
            # Pennylane circuit
            dev = qml.device("default.qubit", wires=n_wires)
            @qml.qnode(dev, interface="torch")
            def circuit(x):
                for i in range(n_wires):
                    qml.RX(x[i], wires=i)
                for i in range(n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                return qml.expval(qml.PauliZ(wires=range(n_wires)))
            self.circuit = circuit
        elif backend == "qiskit":
            # Qiskit circuit placeholder
            self.qc = qiskit.QuantumCircuit(n_wires)
            for i in range(n_wires):
                self.qc.rx(0.0, i)
            for i in range(n_wires - 1):
                self.qc.cx(i, i + 1)
            self.qc.measure_all()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the quantum circuit and return a measurement vector of shape (batch, n_wires).
        """
        if self.backend == "torchquantum":
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)
        elif self.backend == "pennylane":
            return self.circuit(x)
        elif self.backend == "qiskit":
            # Simplified placeholder: return zeros
            return torch.zeros((x.shape[0], self.n_wires), device=x.device, dtype=torch.float32)
        else:
            raise RuntimeError("Quantum backend not initialised.")


class QLSTM__(nn.Module):
    """
    Quantum‑LSTM cell where each gate is implemented by a small variational
    quantum circuit.  The linear embeddings convert the concatenated input and
    hidden state into rotation angles for the circuit.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        backend: str = "torchquantum",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.backend = backend

        self.forget = QLayer(n_qubits, backend=backend)
        self.input = QLayer(n_qubits, backend=backend)
        self.update = QLayer(n_qubits, backend=backend)
        self.output = QLayer(n_qubits, backend=backend)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).
        states : tuple, optional
            Tuple of (h_0, c_0), each of shape (batch, hidden_dim).

        Returns
        -------
        outputs : torch.Tensor
            Hidden states for each time step, shape (seq_len, batch, hidden_dim).
        final_state : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        seq_len, batch, _ = inputs.shape
        for t in range(seq_len):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)

            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger__(nn.Module):
    """
    Sequence tagging model that uses the quantum‑LSTM cell.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        backend: str = "torchquantum",
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM__(embedding_dim, hidden_dim, n_qubits=n_qubits, backend=backend)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len, batch) containing word indices.

        Returns
        -------
        tag_scores : torch.Tensor
            Log‑softmax scores for each tag, shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["QLSTM__", "LSTMTagger__"]
