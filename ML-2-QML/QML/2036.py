import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLayerWrapper(nn.Module):
    """Quantum gate wrapper using a variational circuit implemented with PennyLane.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    output_dim : int
        Dimensionality of the output vector (typically hidden_dim).
    depth : int, default 1
        Number of times the parameterised circuit is repeated.
    seed : int, default 42
        Random seed for initializing parameters.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.wires = list(range(max(input_dim, output_dim)))
        # Use PennyLane's default qubit device
        self.dev = qml.device("default.qubit", wires=self.wires)
        torch.manual_seed(seed)
        # Parameters: shape (depth, input_dim)
        self.params = nn.Parameter(torch.randn(depth, input_dim))
        # qnode with Torch interface
        self.qnode = qml.qnode(self._circuit, interface="torch", diff_method="backprop")

    def _circuit(self, x, params):
        # x: (batch, input_dim)
        for i in range(self.depth):
            for wire in range(self.input_dim):
                qml.RX(params[i, wire], wires=wire)
            # Entangle adjacent wires
            for wire in range(self.input_dim - 1):
                qml.CNOT(wires=[wire, wire + 1])
        # Return expectation values of PauliZ on the first ``output_dim`` wires
        return [qml.expval(qml.PauliZ(w)) for w in range(self.output_dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x, self.params)

class QLSTM(nn.Module):
    """Hybrid LSTM using PennyLane quantum gates for the gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.dropout = dropout

        use_quantum = n_qubits > 0
        gate_dim = hidden_dim

        if use_quantum:
            self.forget_gate = QLayerWrapper(input_dim + hidden_dim, gate_dim, depth=depth)
            self.input_gate = QLayerWrapper(input_dim + hidden_dim, gate_dim, depth=depth)
            self.update_gate = QLayerWrapper(input_dim + hidden_dim, gate_dim, depth=depth)
            self.output_gate = QLayerWrapper(input_dim + hidden_dim, gate_dim, depth=depth)
        else:
            self.forget_gate = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input_gate = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update_gate = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output_gate = nn.Linear(input_dim + hidden_dim, gate_dim)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                depth=depth,
                dropout=dropout,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
