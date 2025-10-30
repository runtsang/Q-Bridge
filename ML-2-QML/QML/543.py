import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QLSTM(nn.Module):
    """
    Hybrid quantum‑classical LSTM. Each gate is a variational quantum circuit
    producing a scalar per sample; the scalar is projected to the full hidden size
    and combined with a classical residual transform.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, circuit_depth: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth

        # Device
        self.qdev = qml.device("default.qubit", wires=n_qubits)

        # Linear layers to feed into quantum circuits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Projection from quantum output (scalar) to hidden_dim
        self.forget_proj = nn.Linear(1, hidden_dim)
        self.input_proj = nn.Linear(1, hidden_dim)
        self.update_proj = nn.Linear(1, hidden_dim)
        self.output_proj = nn.Linear(1, hidden_dim)

        # Classical residual transforms
        self.forget_res = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_res = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_res = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_res = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Quantum gates
        self.forget_gate = self._build_gate()
        self.input_gate = self._build_gate()
        self.update_gate = self._build_gate()
        self.output_gate = self._build_gate()

    def _build_gate(self):
        """Return a PennyLane QNode that takes a (batch, n_qubits) parameter tensor."""
        @qml.qnode(self.qdev, interface="torch")
        def circuit(params):
            for i in range(self.n_qubits):
                qml.RX(params[i], wires=i)
            # Entangle neighbours
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(wires=0))
        return circuit

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_q = self.forget_gate(self.forget_linear(combined)).unsqueeze(-1)
            i_q = self.input_gate(self.input_linear(combined)).unsqueeze(-1)
            g_q = self.update_gate(self.update_linear(combined)).unsqueeze(-1)
            o_q = self.output_gate(self.output_linear(combined)).unsqueeze(-1)

            # Project quantum outputs to hidden_dim
            f = torch.sigmoid(self.forget_proj(f_q))
            i = torch.sigmoid(self.input_proj(i_q))
            g = torch.tanh(self.update_proj(g_q))
            o = torch.sigmoid(self.output_proj(o_q))

            # Classical residuals
            f = f + self.forget_res(hx)
            i = i + self.input_res(hx)
            g = g + self.update_res(hx)
            o = o + self.output_res(hx)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the hybrid QLSTM and a classical LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, circuit_depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, circuit_depth=circuit_depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
