import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """
    Quantum‑classical LSTM cell that replaces the all‑gate quantum
    implementation with controlled‑depth variational circuits.
    The circuit is executed on a simulator via Pennylane, and the
    output of each gate is obtained from a single measurement per gate.
    """
    class QGate(nn.Module):
        """
        Variational gate that uses a circuit on ``n_qubits`` wires.
        The depth of the circuit is controlled by ``depth``.
        """
        def __init__(self, n_qubits: int, depth: int = 1):
            super().__init__()
            self.n_qubits = n_qubits
            self.depth = depth
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(n_qubits)) for _ in range(depth)]
            )
            self.device = qml.device("default.qubit", wires=n_qubits)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: Tensor of shape (batch, n_qubits)
            Returns: Tensor of shape (batch, n_qubits)
            """
            @qml.qnode(self.device, interface="torch")
            def circuit(x_batch, *params):
                for i in range(self.n_qubits):
                    qml.RX(x_batch[i], wires=i)
                for p in params:
                    for i in range(self.n_qubits):
                        qml.RY(p[i], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            return circuit(x, *self.params)

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int, depth: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget_gate = self.QGate(n_qubits, depth)
        self.input_gate = self.QGate(n_qubits, depth)
        self.update_gate = self.QGate(n_qubits, depth)
        self.output_gate = self.QGate(n_qubits, depth)

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.depth_penalty = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32), requires_grad=False
        )

    def forward(self, inputs: torch.Tensor,
                states: tuple | None = None) -> tuple:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple | None = None) -> tuple:
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
    Sequence tagging model that can switch between the classical
    LSTM and the quantum‑classical LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
