import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, Optional

# Define a small variational circuit that will serve as the quantum gate
def variational_layer(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Parameterised single‑qubit rotation followed by a CNOT ladder."""
    n_qubits = len(weights)
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w):
        # Encode input into X‑basis rotations
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
        # Parameterised rotation layer
        for i in range(n_qubits):
            qml.RZ(w[i], wires=i)
            qml.T(wires=i)  # Added T gate
        # Entangling CNOT ladder
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Final entanglement between last and first qubit
        qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.expval(qml.PauliZ(0))
    return circuit(x, weights)

class QLayer(nn.Module):
    """
    Quantum gate implemented with Pennylane.  It maps an input vector
    to a real‑valued scalar via a parameterised rotation followed by
    a Pauli‑Z measurement on qubit 0.  The circuit can be expanded
    to more qubits by increasing the weight matrix size.
    """
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Each qubit gets its own rotation angle
        self.weights = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        return variational_layer(x, self.weights)

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell that replaces the classical gates
    with Pennylane variational layers.  The cell works on batches
    and returns a quantum‑processed hidden state.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum layers for each gate
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Classical linear pre‑processing to match the qubit count
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
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
    """
    Sequence tagging model that can switch between a classical
    LSTM, a quantum‑enhanced LSTM, or a hybrid training mode.
    The quantum path uses Pennylane to compute gradients with
    the same interface as PyTorch.
    """
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
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        # Reshape for (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
