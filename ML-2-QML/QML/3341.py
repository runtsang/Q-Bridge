import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# ----------------------------------------------------------------------
# Quantum fully‑connected layer (torchquantum implementation)
# ----------------------------------------------------------------------
class QFCL(tq.QuantumModule):
    """
    Parameterised quantum circuit acting as a linear layer.
    The circuit encodes each feature into an RX gate and
    applies trainable RX rotations before measuring Z.
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        # Encoder: map classical features to quantum rotations
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_features)]
        )
        # Trainable parameters for each wire
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_features)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        qdev = tq.QuantumDevice(n_wires=self.n_features, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, p in enumerate(self.params):
            tq.RX(p, wires=[i])(qdev)
        # Return expectation value per batch
        return self.measure(qdev).mean(dim=1, keepdim=True)

# ----------------------------------------------------------------------
# Hybrid quantum LSTM
# ----------------------------------------------------------------------
class HybridQLSTM(tq.QuantumModule):
    """
    LSTM cell where each gate is a small quantum circuit
    implemented with torchquantum.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_size = input_dim + hidden_dim

        # Quantum linear layers for each gate
        self.forget_gate = QFCL(gate_size)
        self.input_gate = QFCL(gate_size)
        self.update_gate = QFCL(gate_size)
        self.output_gate = QFCL(gate_size)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)  # (1, gate_size)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

# ----------------------------------------------------------------------
# Tagging model using the hybrid quantum LSTM
# ----------------------------------------------------------------------
class HybridLSTMTagger(tq.QuantumModule):
    """Sequence tagging model that can switch between a classical
    nn.LSTM and the hybrid quantum LSTM."""
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embed_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
