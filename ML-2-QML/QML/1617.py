"""Quantum‑enhanced LSTM with tunable quantum influence and pre‑processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as pnp


class QuantumGate(nn.Module):
    """
    Variational circuit that maps an input vector of size n_wires to
    a vector of expectation values.  The circuit is a single layer
    of RX rotations followed by a single layer of RZ rotations
    and a chain of CNOT gates.  The expectation values are returned
    as a tensor of shape (batch, n_wires).
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Parameters for the RZ rotations
        self.rz_params = nn.Parameter(torch.randn(n_wires))
        self.dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, rz_params):
            for i in range(n_wires):
                qml.RX(inputs[i], wires=i)
                qml.RZ(rz_params[i], wires=i)
            for i in range(n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        self.circuit = circuit

    def forward(self, inputs):
        """
        inputs: Tensor of shape (batch, n_wires)
        Returns: Tensor of shape (batch, n_wires)
        """
        batch = inputs.shape[0]
        outputs = []
        for i in range(batch):
            outputs.append(self.circuit(inputs[i], self.rz_params))
        return torch.stack(outputs)


class QLSTM(nn.Module):
    """
    LSTM cell where gates are realised by a hybrid of classical
    linear layers and a small quantum circuit.  The parameter
    ``alpha`` controls the mixing ratio.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.alpha = alpha

        # Pre‑processing MLP that maps the concatenated input/hidden
        # representation to the quantum‑input dimension.
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits * 4),  # 4 gates
        )

        # Classical linear layers (output a scalar per gate)
        self.forget_linear = nn.Linear(input_dim + hidden_dim, 1)
        self.input_linear = nn.Linear(input_dim + hidden_dim, 1)
        self.update_linear = nn.Linear(input_dim + hidden_dim, 1)
        self.output_linear = nn.Linear(input_dim + hidden_dim, 1)

        # Quantum gate modules
        self.forget_gate = QuantumGate(n_qubits)
        self.input_gate = QuantumGate(n_qubits)
        self.update_gate = QuantumGate(n_qubits)
        self.output_gate = QuantumGate(n_qubits)

    def forward(self, inputs, states=None):
        """
        inputs: Tensor of shape (seq_len, batch, input_dim)
        states: Tuple of (hx, cx) each of shape (batch, hidden_dim)
        Returns: outputs (seq_len, batch, hidden_dim) and final states
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)  # (batch, input_dim + hidden_dim)
            # Classical gate outputs (batch, 1)
            f_c = torch.sigmoid(self.forget_linear(combined))
            i_c = torch.sigmoid(self.input_linear(combined))
            g_c = torch.tanh(self.update_linear(combined))
            o_c = torch.sigmoid(self.output_linear(combined))

            # Quantum gate outputs (batch, 1) --> mean over qubits
            q_inputs = self.preprocess(combined)  # (batch, n_qubits * 4)
            q_forget = torch.mean(self.forget_gate(q_inputs[:, :self.n_qubits]), dim=1, keepdim=True)
            q_input = torch.mean(self.input_gate(q_inputs[:, self.n_qubits: 2 * self.n_qubits]), dim=1, keepdim=True)
            q_update = torch.mean(self.update_gate(q_inputs[:, 2 * self.n_qubits: 3 * self.n_qubits]), dim=1, keepdim=True)
            q_output = torch.mean(self.output_gate(q_inputs[:, 3 * self.n_qubits:]), dim=1, keepdim=True)

            # Mix classical and quantum gates
            f = (1 - self.alpha) * f_c + self.alpha * torch.sigmoid(q_forget)
            i = (1 - self.alpha) * i_c + self.alpha * torch.sigmoid(q_input)
            g = (1 - self.alpha) * g_c + self.alpha * torch.tanh(q_update)
            o = (1 - self.alpha) * o_c + self.alpha * torch.sigmoid(q_output)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
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
        alpha: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, alpha=alpha)
        else:
            # Classic LSTM with dropout
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
            self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        """
        sentence: Tensor of shape (seq_len, batch)
        Returns: Tensor of shape (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, QLSTM):
            lstm_out = self.lstm(embeds)[0]
        else:
            lstm_out, _ = self.lstm(embeds)
            lstm_out = self.dropout(lstm_out)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["QLSTM", "LSTMTagger"]
