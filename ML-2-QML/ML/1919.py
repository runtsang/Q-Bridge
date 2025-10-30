import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumGate(nn.Module):
    """
    Simple variational quantum circuit used as a gate.
    The circuit encodes the input with RX gates, then applies
    a few layers of parameterised RY rotations and CNOTs.
    """
    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers * n_qubits))
        self.qnode = qml.qnode(self.dev, interface="torch", diff_method="backprop")(self.circuit)

    def circuit(self, params, x):
        # Encode input
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
        # Variational layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                idx = layer * self.n_qubits + i
                qml.RY(params[idx], wires=i)
            # Entanglement pattern
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        # x: (batch, n_qubits)
        batch = x.shape[0]
        out = []
        for i in range(batch):
            out.append(self.qnode(self.params, x[i]))
        return torch.stack(out, dim=0)

class HybridQLSTM(nn.Module):
    """
    Classical LSTM with optional quantum gates per LSTM gate.
    Each gate can be either a linear layer or a variational quantum circuit.
    Dropout is applied to the hidden state after each step.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        use_quantum: bool = False,
        quantum_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.dropout = dropout

        gate_dim = hidden_dim

        # Linear projections for gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, gate_dim)

        if use_quantum:
            # Quantum gates
            self.forget_gate = QuantumGate(n_qubits, quantum_layers)
            self.input_gate = QuantumGate(n_qubits, quantum_layers)
            self.update_gate = QuantumGate(n_qubits, quantum_layers)
            self.output_gate = QuantumGate(n_qubits, quantum_layers)
        else:
            # Classical gates
            self.forget_gate = nn.Linear(gate_dim, gate_dim)
            self.input_gate = nn.Linear(gate_dim, gate_dim)
            self.update_gate = nn.Linear(gate_dim, gate_dim)
            self.output_gate = nn.Linear(gate_dim, gate_dim)

    def forward(self, inputs: torch.Tensor, states: torch.Tensor | None = None):
        """
        inputs: (seq_len, batch, input_dim)
        states: (hx, cx) each of shape (batch, hidden_dim)
        """
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
            if self.dropout > 0.0:
                hx = F.dropout(hx, p=self.dropout, training=self.training)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical,
    hybrid, or fully quantum LSTM via the `use_quantum` flag.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum: bool = False,
        quantum_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0 and use_quantum:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits,
                use_quantum=True,
                quantum_layers=quantum_layers,
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

__all__ = ["HybridQLSTM", "LSTMTagger"]
