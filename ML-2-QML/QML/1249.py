import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional


class QLayer(nn.Module):
    """
    Variational quantum layer implemented with PennyLane.
    Each gate in the LSTM uses a shallow circuit of depth `depth`.
    """
    def __init__(self, n_wires: int, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_wires)
        # Trainable variational parameters
        self.variational_params = nn.Parameter(
            torch.randn(depth * n_wires)
        )
        # Pre‑define the QNode
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

    def circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, n_wires)
        for i in range(self.n_wires):
            qml.RX(inputs[:, i], wires=i)
        for d in range(self.depth):
            for w in range(self.n_wires):
                qml.RY(params[d * self.n_wires + w], wires=w)
            # Entanglement pattern
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
            qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_wires)
        return self.qnode(x, self.variational_params)


class QLSTM(nn.Module):
    """
    Quantum LSTM cell where each gate is a variational quantum circuit.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 depth: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Quantum gates
        self.forget = QLayer(n_qubits, depth)
        self.input = QLayer(n_qubits, depth)
        self.update = QLayer(n_qubits, depth)
        self.output = QLayer(n_qubits, depth)

        # Classical linear projections to the qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=-1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the quantum LSTM and a vanilla nn.LSTM.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 depth: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
