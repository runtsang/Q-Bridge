import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple

class QLayer(nn.Module):
    """
    Variational quantum layer that maps classical input features to a quantum measurement vector.
    """
    def __init__(self, input_dim: int, n_qubits: int, n_params: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.proj = nn.Linear(input_dim, n_qubits)
        # Initialise trainable parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(n_qubits, n_params))
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, params):
            # Encode inputs via RX rotations
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            # Parameterized rotations
            for i in range(n_qubits):
                for j in range(n_params):
                    qml.RY(params[i, j], wires=i)
            # Entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation values of PauliZ
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, input_dim).

        Returns
        -------
        out : torch.Tensor
            Tensor of shape (batch, n_qubits).
        """
        batch_size = x.shape[0]
        x_proj = self.proj(x)  # shape (batch, n_qubits)
        # Apply circuit to each sample
        outputs = torch.stack([self.circuit(x_proj[i], self.params) for i in range(batch_size)], dim=0)
        return outputs

class QRegulariser(nn.Module):
    """
    Quantum regulariser that measures the fidelity between a hidden state and a target vector.
    """
    def __init__(self, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.proj = nn.Linear(hidden_dim, n_qubits)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs):
            # Encode hidden state
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            # Measure fidelity with |0...0>
            return qml.probs(wires=range(n_qubits))[0]  # probability of all zeros

        self.circuit = circuit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute a scalar regularisation term based on the fidelity of the hidden state.
        """
        batch_size = hidden_state.shape[0]
        hidden_proj = self.proj(hidden_state)  # (batch, n_qubits)
        fidelities = torch.stack([self.circuit(hidden_proj[i]) for i in range(batch_size)], dim=0)
        # We want a penalty: higher fidelity => lower penalty
        reg = 1.0 - fidelities
        return reg.mean()

class QLSTMEnhancedQuantum(nn.Module):
    """
    Quantum LSTM cell where each gate is realised by a variational quantum circuit.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
        residual: bool = True,
        reg_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.residual = residual
        self.dropout = nn.Dropout(dropout)

        # Quantum layers for gates
        combined_dim = input_dim + hidden_dim
        self.forget_layer = QLayer(combined_dim, n_qubits)
        self.input_layer = QLayer(combined_dim, n_qubits)
        self.update_layer = QLayer(combined_dim, n_qubits)
        self.output_layer = QLayer(combined_dim, n_qubits)

        # Linear projections to match qubit count
        self.linear_forget = nn.Linear(combined_dim, n_qubits)
        self.linear_input = nn.Linear(combined_dim, n_qubits)
        self.linear_update = nn.Linear(combined_dim, n_qubits)
        self.linear_output = nn.Linear(combined_dim, n_qubits)

        self.reg_func = reg_func if reg_func is not None else QRegulariser(hidden_dim, n_qubits)

    def forward(
        self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the quantum LSTM cell.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).
        states : Tuple[torch.Tensor, torch.Tensor] | None
            Tuple of hidden and cell states (hx, cx). If None, initialise to zeros.

        Returns
        -------
        outputs : torch.Tensor
            Hidden states of shape (batch, seq_len, hidden_dim).
        reg_term : torch.Tensor
            Quantum regularisation term.
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        if states is not None:
            hx, cx = states

        outputs = []
        reg_terms = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            combined = torch.cat([x_t, hx], dim=1)
            # Gate computations via quantum layers
            f = torch.sigmoid(self.forget_layer(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_layer(self.linear_input(combined)))
            g = torch.tanh(self.update_layer(self.linear_update(combined)))
            o = torch.sigmoid(self.output_layer(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)
            if self.residual:
                res = self.linear_forget(combined)  # placeholder residual
                hx = hx + res

            outputs.append(hx.unsqueeze(1))
            reg_terms.append(self.reg_func(hx))

        outputs = torch.cat(outputs, dim=1)
        reg_term = torch.stack(reg_terms, dim=1).mean(dim=1)  # mean over time steps
        return outputs, reg_term

class QLSTMTaggerEnhanced(nn.Module):
    """
    Quantum‑enhanced LSTM tagger that replaces each gate with a variational quantum circuit.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        dropout: float = 0.1,
        residual: bool = True,
        reg_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Quantum LSTM
        self.lstm = QLSTMEnhancedQuantum(
            embedding_dim, hidden_dim, n_qubits, dropout=dropout, residual=residual, reg_func=reg_func
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Input sentence as a LongTensor of shape (batch, seq_len).

        Returns
        -------
        log_probs : torch.Tensor
            Log-probabilities for each token.
        reg_term : torch.Tensor
            Quantum regularisation term.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, reg_term = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, reg_term

__all__ = ["QLSTMTaggerEnhanced"]
