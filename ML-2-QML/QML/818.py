import torch
import torch.nn as nn
import pennylane as qml

class QuantumGate(nn.Module):
    """
    A parameterised quantum circuit that outputs a scalar activation.
    The circuit consists of a chain of RX rotations followed by CNOT
    entangling gates.  The depth of the circuit is configurable.
    """

    def __init__(self, hidden_dim: int, n_qubits: int, depth: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.linear = nn.Linear(hidden_dim, n_qubits)
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, params: torch.Tensor):
        # params shape: (depth * n_qubits,)
        idx = 0
        for d in range(self.depth):
            for w in range(self.n_qubits):
                qml.RX(params[idx], wires=w)
                idx += 1
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, hidden_dim)
        params = self.linear(x)  # (batch, n_qubits)
        # expand to depth
        params = params.unsqueeze(1).repeat(1, self.depth, 1)  # (batch, depth, n_qubits)
        params = params.view(-1, self.depth * self.n_qubits)  # (batch*depth, depth*n_qubits)
        outputs = []
        for p in params:
            outputs.append(self.qnode(p))
        return torch.stack(outputs)

class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell.  Each gate is realised by a
    parameterised variational quantum circuit.  The classical linear
    projection reduces the input dimensionality before feeding it into
    the quantum circuit, keeping the overall model size comparable to
    a pure classical LSTM.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int, depth: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Quantum gates
        self.forget = QuantumGate(hidden_dim, n_qubits, depth)
        self.input = QuantumGate(hidden_dim, n_qubits, depth)
        self.update = QuantumGate(hidden_dim, n_qubits, depth)
        self.output = QuantumGate(hidden_dim, n_qubits, depth)

        # Linear projections to feed into the quantum circuits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.forget_linear(combined)))
            i = torch.sigmoid(self.input(self.input_linear(combined)))
            g = torch.tanh(self.update(self.update_linear(combined)))
            o = torch.sigmoid(self.output(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses :class:`HybridQLSTM` for the
    recurrent layer.  The tagger is identical to the original seed but
    now incorporates quantum‑parameterised gates.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int, depth: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits, depth)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
