import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class _QuantumGate(nn.Module):
    """Variational quantum circuit that encodes a classical vector into qubit rotations,
    applies a depth‑controlled parameterized circuit, and returns expectation values.
    Assumes hidden_dim == n_qubits for simplicity."""
    def __init__(self, n_qubits: int, depth: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(depth, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Encode input vector into RX rotations
            for i in range(self.n_qubits):
                qml.RX(x[:, i], wires=i)
            # Variational layers
            for layer in range(self.depth):
                for i in range(self.n_qubits):
                    qml.RZ(params[layer, i, 0], wires=i)
                    qml.RX(params[layer, i, 1], wires=i)
                    qml.RY(params[layer, i, 2], wires=i)
                # Entangling CNOT chain
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Measure expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x, self.params)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM where each gate is processed by a variational quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 depth: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.forget_qc = _QuantumGate(n_qubits, depth)
        self.input_qc = _QuantumGate(n_qubits, depth)
        self.update_qc = _QuantumGate(n_qubits, depth)
        self.output_qc = _QuantumGate(n_qubits, depth)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))

            f = self.forget_qc(f)
            i = self.input_qc(i)
            g = self.update_qc(g)
            o = self.output_qc(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits,
                              depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
