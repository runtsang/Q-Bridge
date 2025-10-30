import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple, Optional

class QuantumGate(nn.Module):
    """
    Variational quantum circuit that transforms a vector of
    length ``n_qubits`` into a new vector of the same length.
    The circuit consists of:
        - RX encoding of the input features.
        - Two layers of parametrised rotations (RY) and entangling
          CNOT gates.
        - Measurement of the Pauli‑Z expectation on each wire.
    The parameters are trainable and shared across all gates
    that instantiate the same ``QuantumGate``.
    """

    def __init__(self, n_qubits: int, n_params: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.wires = list(range(n_qubits))

        # PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=self.wires)

        # Initialise trainable parameters
        self.params = nn.Parameter(torch.randn(n_params))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode input features
            for i, wire in enumerate(self.wires):
                qml.RX(inputs[i], wires=wire)

            # Variational layers
            param_iter = iter(params)
            for _ in range(2):
                for wire in self.wires:
                    qml.RY(next(param_iter), wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])

            # Return expectation values of Pauli‑Z on each wire
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, n_qubits)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, n_qubits)``.
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.circuit(x[i], self.params))
        return torch.stack(outputs)


class QLSTMEnhanced(nn.Module):
    """
    Quantum‑enhanced LSTM cell that replaces each classical gate
    with a variational quantum circuit.  The architecture mirrors
    the original ``QLSTM`` interface but adds a dropout layer after
    the quantum output to regularise the gate activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim == n_qubits, (
            "For quantum gates, hidden_dim must equal n_qubits"
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Quantum gates
        self.forget = QuantumGate(n_qubits, n_qubits * 2)
        self.input = QuantumGate(n_qubits, n_qubits * 2)
        self.update = QuantumGate(n_qubits, n_qubits * 2)
        self.output = QuantumGate(n_qubits, n_qubits * 2)

        # Linear projections into the quantum space
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

            # Drop‑out on gate activations
            f = self.dropout(f)
            i = self.dropout(i)
            g = self.dropout(g)
            o = self.dropout(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        output_tensor = torch.cat(outputs, dim=0)
        return output_tensor, (hx, cx)

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


class LSTMTaggerEnhanced(nn.Module):
    """
    Sequence‑tagging model that can switch between the enhanced quantum
    LSTM cell and the standard ``nn.LSTM``.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMEnhanced(
                embedding_dim, hidden_dim, n_qubits=n_qubits, dropout=dropout
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMEnhanced", "LSTMTaggerEnhanced"]
