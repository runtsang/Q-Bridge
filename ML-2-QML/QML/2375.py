"""
HybridEstimator module with quantum support for regression and tagging.
The quantum branch replaces the classical layers with parameter‑efficient
quantum circuits.  The implementation uses torchquantum for differentiable
simulation, but can also be executed on a real device via the
qiskit/primitives interface if desired.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridEstimator(nn.Module):
    """Hybrid estimator with quantum and classical branches.

    Parameters
    ----------
    mode : {"regression", "tagging"}
        Which task to perform.
    use_q : bool, optional
        If True the quantum implementation is used; otherwise a
        classical fallback is used.
    n_qubits : int, optional
        Number of qubits for the quantum implementation.
    embedding_dim : int, optional
        Embedding size for the tagging task.
    hidden_dim : int, optional
        Hidden size for the LSTM.
    vocab_size : int, optional
        Vocabulary size for the embedding layer.
    tagset_size : int, optional
        Number of tags for the tagging task.
    """

    def __init__(
        self,
        mode: str = "regression",
        use_q: bool = True,
        n_qubits: int = 4,
        embedding_dim: int = 50,
        hidden_dim: int = 100,
        vocab_size: int = 10000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.use_q = use_q
        self.n_qubits = n_qubits

        if mode == "regression":
            self.regressor = (
                _QuantumRegressor(n_qubits) if use_q else _ClassicalRegressor()
            )
        elif mode == "tagging":
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            if use_q:
                self.lstm = _QuantumLSTM(embedding_dim, hidden_dim, n_qubits)
            else:
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "regression":
            return self.regressor(inputs)
        else:
            embeds = self.word_embeddings(inputs)
            lstm_out, _ = self.lstm(embeds)
            tag_logits = self.hidden2tag(lstm_out)
            return F.log_softmax(tag_logits, dim=-1)

class _QuantumRegressor(tq.QuantumModule):
    """Parameter‑efficient quantum feed‑forward regressor."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Encode each input feature into a separate qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
            ]
        )
        # Trainable rotation gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Return expectation value of the first qubit as scalar output
        return self.measure(qdev)[:, 0:1]

class _QuantumLSTM(nn.Module):
    """LSTM where each gate is realised by a small quantum circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_wires)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Use the first qubit's expectation as gate output
            return self.measure(qdev)[:, 0:1]

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to quantum gate dimension
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum layers for each gate
        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
        # inputs shape: (seq_len, batch, input_dim)
        batch_size = inputs.size(1)
        device = inputs.device
        if states is None:
            hx = torch.zeros(batch_size, self.hidden_dim, device=device)
            cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            hx, cx = states

        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class _ClassicalRegressor(nn.Module):
    """Fallback classical regressor used when quantum mode is disabled."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def EstimatorQNN() -> HybridEstimator:
    """Factory that returns a HybridEstimator instance (default quantum)."""
    return HybridEstimator()

__all__ = ["HybridEstimator", "EstimatorQNN"]
