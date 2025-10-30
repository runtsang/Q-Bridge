import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QGate(nn.Module):
    """
    Variational quantum circuit used as an LSTM gate.  The circuit consists of
    parameterized RX rotations on each qubit followed by a chain of CNOTs to
    entangle the wires.  The output is the expectation value of PauliZ on
    all qubits, summed and normalized to [0,1].
    """
    def __init__(self, n_qubits: int, device: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=0)

        # Parameters for each RX gate
        self.params = nn.Parameter(torch.randn(n_qubits))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor):
            # Encode the classical input into rotation angles
            for i in range(self.n_qubits):
                qml.RX(x[:, i], wires=i)
            # Trainable rotations
            for i in range(self.n_qubits):
                qml.RX(self.params[i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a batch of qubit rotations encoded as a tensor of shape (batch, n_qubits)
        out = self.circuit(x)
        # Convert list to tensor
        out = torch.stack(out, dim=1)
        # Map to [0,1] and sum to obtain a scalar gate value per batch element
        return torch.sigmoid(-torch.sum(out, dim=1))

class QLSTMPlus(nn.Module):
    """
    Quantum-enhanced LSTM where each gate is a variational quantum circuit.
    The module supports a learnable noise parameter `beta` that scales the
    output of the quantum circuit, allowing the network to interpolate
    between classical and quantum regimes.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        device: str = "default.qubit",
        noise_beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.noise_beta = nn.Parameter(torch.tensor(noise_beta))

        if n_qubits == 0:
            # Pure classical fallback
            self.lstm = nn.LSTM(input_dim, hidden_dim)
        else:
            # Linear layers that map the concatenated input+state to qubit rotations
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

            # Quantum gates
            self.qgate_forget = QGate(n_qubits, device)
            self.qgate_input = QGate(n_qubits, device)
            self.qgate_update = QGate(n_qubits, device)
            self.qgate_output = QGate(n_qubits, device)

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

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.n_qubits == 0:
            return self.lstm(inputs, states)

        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = self.qgate_forget(self.linear_forget(combined)).unsqueeze(-1)
            i = self.qgate_input(self.linear_input(combined)).unsqueeze(-1)
            g = self.qgate_update(self.linear_update(combined)).unsqueeze(-1)
            o = self.qgate_output(self.linear_output(combined)).unsqueeze(-1)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Apply learnable noise
            hx = hx + self.noise_beta * torch.randn_like(hx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses the hybrid QLSTMPlus.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=0, device="default.qubit"):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMPlus(embedding_dim, hidden_dim, n_qubits, device=device)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.reshape(-1, self.lstm.hidden_dim))
        tag_logits = tag_logits.reshape(lstm_out.size(0), lstm_out.size(1), -1)
        return F.log_softmax(tag_logits, dim=2)
