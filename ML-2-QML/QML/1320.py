import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class _QuantumGate(nn.Module):
    """
    Small variational quantum circuit that maps a feature vector to a qubit state.
    The circuit consists of a trainable rotation layer followed by a fixed entangling pattern.
    """
    def __init__(self, n_qubits: int, feature_dim: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        # Linear layer to map features to rotation angles for each qubit
        self.angle_mapper = nn.Linear(feature_dim, n_qubits * 3)  # 3 angles per qubit: Rx, Ry, Rz
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            # params shape: (n_qubits, 3)
            for q in range(self.n_qubits):
                qml.RX(params[q, 0], wires=q)
                qml.RY(params[q, 1], wires=q)
                qml.RZ(params[q, 2], wires=q)
            # Entangling layer: CNOT chain
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            # Measure expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(wires=q)) for q in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, feature_dim).
        Returns
        -------
        out : torch.Tensor
            Quantum measurement results of shape (batch, n_qubits).
        """
        angles = self.angle_mapper(x)  # shape: (batch, n_qubits*3)
        angles = angles.view(-1, self.n_qubits, 3)  # shape: (batch, n_qubits, 3)
        # Execute circuit for each batch element
        out = torch.stack([self.circuit(params) for params in angles], dim=0)
        return out  # shape: (batch, n_qubits)

class HybridQLSTM(nn.Module):
    """
    Quantum‑augmented LSTM cell.
    Each gate is implemented as a small variational quantum circuit.
    The cell can be used in a pure‑quantum mode or as a hybrid with classical linear layers.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 mode: str = "quantum") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.mode = mode.lower()
        if self.mode not in {"quantum", "hybrid"}:
            raise ValueError("mode must be 'quantum' or 'hybrid' in the quantum implementation")

        # Gate circuits
        self.forget_gate = _QuantumGate(n_qubits, input_dim + hidden_dim)
        self.input_gate = _QuantumGate(n_qubits, input_dim + hidden_dim)
        self.update_gate = _QuantumGate(n_qubits, input_dim + hidden_dim)
        self.output_gate = _QuantumGate(n_qubits, input_dim + hidden_dim)

        # Optional linear layers for hybrid mode
        if self.mode == "hybrid":
            self.fc_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.fc_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.fc_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.fc_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Quantum forward pass of the LSTM cell.
        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (seq_len, batch, input_dim).
        states : tuple, optional
            Initial hidden and cell states (h, c). If None, zeros are used.
        Returns
        -------
        outputs : torch.Tensor
            Sequence of hidden states of shape (seq_len, batch, hidden_dim).
        final_states : tuple
            Final hidden and cell states (h, c).
        """
        batch_size = inputs.size(1)
        device = inputs.device
        h, c = self._init_state(batch_size, device) if states is None else states

        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)  # shape: (batch, input_dim + hidden_dim)
            if self.mode == "quantum":
                f = torch.sigmoid(self.forget_gate(combined))
                i = torch.sigmoid(self.input_gate(combined))
                g = torch.tanh(self.update_gate(combined))
                o = torch.sigmoid(self.output_gate(combined))
            else:  # hybrid
                f = torch.sigmoid(self.forget_gate(combined))
                i = torch.sigmoid(self.input_gate(combined))
                g = torch.tanh(self.update_gate(combined))
                o = torch.sigmoid(self.output_gate(combined))
                # Merge quantum outputs with linear projections
                f = torch.sigmoid(self.fc_forget(combined) + f)
                i = torch.sigmoid(self.fc_input(combined) + i)
                g = torch.tanh(self.fc_update(combined) + g)
                o = torch.sigmoid(self.fc_output(combined) + o)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses HybridQLSTM in quantum or hybrid mode.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int,
                 mode: str = "quantum") -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, mode=mode)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["HybridQLSTM", "LSTMTagger"]
