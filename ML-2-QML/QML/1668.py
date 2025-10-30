import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QLayer(nn.Module):
    """
    Variational quantum layer that maps a classical vector to a probability vector.
    The circuit uses a hardware‑efficient ansatz with rotation and CNOT entangling layers.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(0.01 * torch.randn(n_qubits, 3, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the variational circuit.
        Args:
            x: Tensor of shape (batch, n_qubits), values in [-1, 1].
        Returns:
            Tensor of shape (batch, n_qubits) – probabilities of measuring each qubit in state |1⟩.
        """
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            for i in range(self.n_qubits):
                qml.Rot(*params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        probs = circuit(x, self.params)
        return (probs + 1.0) / 2.0

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM where each gate is realised by a QLayer.
    The QLayer outputs a probability vector that is interpreted as a gate activation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: tuple | None = None) -> tuple:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update(self.update_lin(combined)))
            o = torch.sigmoid(self.output(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple | None = None) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use the classical QLSTM or the quantum‑enhanced QLSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
