import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLinearGate(nn.Module):
    """
    Quantum linear gate implemented with a Pennylane QNode.
    Input features are encoded with RY rotations, followed by a
    trainable ansatz of RZ rotations and CNOTs.  The measurement
    returns the expectation values of PauliZ for each qubit, which
    serve as the output feature vector.
    """
    def __init__(self, in_features: int, out_features: int, n_qubits: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev, interface="torch")
        def circuit(x):
            # Encode input
            for i in range(min(self.in_features, self.n_qubits)):
                qml.RY(x[i], wires=i)
            # Ansatz
            for i in range(self.n_qubits):
                qml.RZ(self.weight[i], wires=i)
                if i < self.n_qubits - 1:
                    qml.CNOT(wires=[i, i + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        self.weight = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _ = x.shape
        out = []
        for i in range(batch):
            out.append(self.circuit(x[i]))
        return torch.stack(out)

class QLSTMCell(nn.Module):
    """
    LSTM cell where gates are quantum linear gates.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        self.forget_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, n_qubits)
        self.input_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, n_qubits)
        self.update_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, n_qubits)
        self.output_gate = QLinearGate(input_dim + hidden_dim, hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((x, h_prev), dim=1)
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        g = torch.tanh(self.update_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        h = self.dropout(h)
        return h, c

class QLSTM(nn.Module):
    """
    Multi‑layer quantum‑enhanced LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 n_qubits: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_dim = input_dim if l == 0 else hidden_dim
            self.layers.append(
                QLSTMCell(in_dim, hidden_dim, n_qubits=n_qubits, dropout=dropout)
            )

    def forward(self, inputs: torch.Tensor,
                h_0: Optional[torch.Tensor] = None,
                c_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len, batch_size, _ = inputs.shape
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=inputs.device)
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=inputs.device)

        outputs = []
        h_n = []
        c_n = []

        for t in range(seq_len):
            x = inputs[t]
            h_t = []
            c_t = []
            for l, layer in enumerate(self.layers):
                h, c = layer(x, h_0[l], c_0[l])
                x = h
                h_t.append(h)
                c_t.append(c)
            outputs.append(x.unsqueeze(0))
            h_n.append(torch.stack(h_t))
            c_n.append(torch.stack(c_t))

        outputs = torch.cat(outputs, dim=0)
        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)
        return outputs, (h_n, c_n)

class QLSTMGen(nn.Module):
    """
    Sequence tagging model that uses the quantum‑enhanced LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 num_layers: int = 2, n_qubits: int = 4, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                          n_qubits=n_qubits, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: (seq_len, batch)
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTMGen", "QLSTM", "QLSTMCell", "QLinearGate"]
