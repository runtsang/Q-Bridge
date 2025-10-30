import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM where each gate is implemented by a
    variational circuit on 2*n_qubits wires. The circuit receives a
    linear embedding of the concatenated input and hidden state.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 device: str = 'cpu'):
        super().__init__()
        assert hidden_dim == n_qubits, "For the quantum variant hidden_dim must equal n_qubits."
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.device = device

        # Linear mapping into a 2*n_qubits dimensional vector
        self.linear_q = nn.Linear(input_dim + hidden_dim, 2 * n_qubits)

        # Variational parameters for each gate
        self.forget_params = nn.Parameter(torch.randn(n_qubits))
        self.input_params = nn.Parameter(torch.randn(n_qubits))
        self.update_params = nn.Parameter(torch.randn(n_qubits))
        self.output_params = nn.Parameter(torch.randn(n_qubits))

        # Quantum device
        self.qdev = qml.device("default.qubit", wires=2 * n_qubits)

    def _quantum_gate(self, params: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        """Apply a variational circuit to the quantum register and
        return a vector of expectation values of Pauli‑Z on each wire.
        """
        @qml.qnode(self.qdev, interface="torch")
        def circuit(x, p):
            # Encode input part into first n_qubits
            for i in range(self.n_qubits):
                qml.RX(x[:, i], wires=i)
            # Encode hidden part into next n_qubits
            for i in range(self.n_qubits):
                qml.RX(x[:, self.n_qubits + i], wires=self.n_qubits + i)

            # Variational layers
            for i, angle in enumerate(p):
                qml.RY(angle, wires=i)
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(inp, params)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            qvec = self.linear_q(combined)

            f = torch.sigmoid(self._quantum_gate(self.forget_params, qvec))
            i = torch.sigmoid(self._quantum_gate(self.input_params, qvec))
            g = torch.tanh(self._quantum_gate(self.update_params, qvec))
            o = torch.sigmoid(self._quantum_gate(self.output_params, qvec))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and
    quantum LSTM. The quantum variant uses the QLSTM defined above."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, device: str = 'cpu'):
        super().__init__()
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits, device=device)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.unsqueeze(0).transpose(0, 1))
        return F.log_softmax(self.hidden2tag(lstm_out.squeeze(0)), dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
