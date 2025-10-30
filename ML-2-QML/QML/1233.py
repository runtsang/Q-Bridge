import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell: each gate is a variational quantum circuit."""
    class _QuantumGate(nn.Module):
        def __init__(self, n_qubits: int, depth: int, device: str = "default.qubit"):
            super().__init__()
            self.n_qubits = n_qubits
            self.depth = depth
            self.dev = qml.device(device, wires=n_qubits)
            self.params = nn.Parameter(torch.randn(depth, n_qubits))

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def circuit(inputs, params):
                for i in range(self.n_qubits):
                    qml.RX(inputs[i], wires=i)
                for d in range(self.depth):
                    for i in range(self.n_qubits):
                        qml.RY(params[d, i], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            self.circuit = circuit

        def forward(self, x):
            # x shape: (batch, n_qubits)
            out = []
            for i in range(x.shape[0]):
                out.append(self.circuit(x[i], self.params))
            return torch.stack(out)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.forget_gate = self._QuantumGate(n_qubits, depth)
        self.input_gate = self._QuantumGate(n_qubits, depth)
        self.update_gate = self._QuantumGate(n_qubits, depth)
        self.output_gate = self._QuantumGate(n_qubits, depth)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Tagger that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=0, depth=2, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
