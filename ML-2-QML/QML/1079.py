import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, Optional

class QGate(nn.Module):
    """A small variational circuit that outputs a single expectation value
    used as a gate activation.  The circuit is differentiable on the
    Pennylane device and can be swapped for different ansätze.
    """
    def __init__(self, n_params: int, device: str = "default.qubit"):
        super().__init__()
        self.n_params = n_params
        self.dev = qml.device(device, wires=1)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(params: torch.Tensor):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params shape: (batch, n_params)
        return self.circuit(params)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell that injects quantum‑derived gate values
    via a variational circuit with two parameters per gate.  The circuit
    architecture is modular and can be swapped for different ansätze
    (e.g., hardware‑efficient or basis‑rotation).  The design keeps the
    original interface while adding a small quantum‑to‑classical
    post‑processing step that is differentiable on the device.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 1, device: str = "default.qubit"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.device = device

        # Classical linear layers that map to quantum parameters
        self.linear_forget = nn.Linear(input_dim + hidden_dim, 2 * n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, 2 * n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, 2 * n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, 2 * n_qubits)

        # Quantum gates (one per qubit)
        self.forget_gate = QGate(2 * n_qubits, device=device)
        self.input_gate = QGate(2 * n_qubits, device=device)
        self.update_gate = QGate(2 * n_qubits, device=device)
        self.output_gate = QGate(2 * n_qubits, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : (seq_len, batch, input_dim)
            Input sequence.
        states : (h, c) optional
            Initial hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Map to quantum parameters
            f_params = self.linear_forget(combined)
            i_params = self.linear_input(combined)
            g_params = self.linear_update(combined)
            o_params = self.linear_output(combined)

            # Quantum gate activations
            f = torch.sigmoid(self.forget_gate(f_params))
            i = torch.sigmoid(self.input_gate(i_params))
            g = torch.tanh(self.update_gate(g_params))
            o = torch.sigmoid(self.output_gate(o_params))

            # LSTM update
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
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
