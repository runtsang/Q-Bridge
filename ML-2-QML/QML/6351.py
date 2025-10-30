import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumGate(tq.QuantumModule):
    """
    Variational quantum circuit used as a gate activation. The depth controls
    the number of parameterized layers.
    """
    def __init__(self, n_qubits: int, depth: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        # Parameterized rotation angles for each layer
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(n_qubits, 3)) for _ in range(depth)
        ])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        # Encode input amplitudes into rotation angles
        for i in range(self.n_qubits):
            tq.RX(x[:, i], wires=i)(qdev)
        # Variational layers
        for layer in range(self.depth):
            for q in range(self.n_qubits):
                tq.RY(self.params[layer][q, 0], wires=q)(qdev)
                tq.RZ(self.params[layer][q, 1], wires=q)(qdev)
            # Entangling CNOT chain
            for q in range(self.n_qubits - 1):
                tqf.cnot(qdev, wires=[q, q + 1])
        return self.measure(qdev)

class QLSTMGen(nn.Module):
    """
    Quantum LSTM cell where each gate is a variational quantum circuit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget = QuantumGate(n_qubits, depth)
        self.input = QuantumGate(n_qubits, depth)
        self.update = QuantumGate(n_qubits, depth)
        self.output = QuantumGate(n_qubits, depth)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Classical post‑processing to map qubit outputs to hidden dimension
        self.post = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.post(self.forget(self.linear_forget(combined))))
            i = torch.sigmoid(self.post(self.input(self.linear_input(combined))))
            g = torch.tanh(self.post(self.update(self.linear_update(combined))))
            o = torch.sigmoid(self.post(self.output(self.linear_output(combined))))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTaggerGen(nn.Module):
    """
    Sequence tagging model that can swap between the quantum LSTM and a standard nn.LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits, depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTaggerGen"]
