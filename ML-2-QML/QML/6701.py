import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class QuantumGate(nn.Module):
    """Quantum gate implemented with a Pennylane QNode."""
    def __init__(self, in_features: int, out_features: int, n_qubits: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_qubits = n_qubits

        # Map classical input to a quantum state
        self.input_mapper = nn.Linear(in_features, n_qubits)

        # Trainable rotation angles
        self.theta = nn.Parameter(torch.randn(n_qubits))

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(mapped):
            # encoded input + trainable rotations
            for i in range(n_qubits):
                qml.RX(mapped[..., i] + self.theta[i], wires=i)
            # entangle the qubits
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        mapped = self.input_mapper(x)  # (batch, n_qubits)
        out = self.circuit(mapped)     # (batch, n_qubits)
        # If required, map to desired out_features
        if self.out_features!= self.n_qubits:
            out = nn.functional.linear(out, torch.eye(self.out_features, self.n_qubits, device=out.device))
        return out

class QLSTM__gen558(nn.Module):
    """Quantum‑enhanced LSTM where each gate is a variational quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        if hidden_dim!= n_qubits:
            raise ValueError("For quantum LSTM, hidden_dim must equal n_qubits")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QuantumGate(input_dim + hidden_dim, hidden_dim, n_qubits)
        self.input = QuantumGate(input_dim + hidden_dim, hidden_dim, n_qubits)
        self.update = QuantumGate(input_dim + hidden_dim, hidden_dim, n_qubits)
        self.output = QuantumGate(input_dim + hidden_dim, hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
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
            self.lstm = QLSTM__gen558(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM__gen558", "LSTMTagger"]
