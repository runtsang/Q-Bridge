import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumGate(nn.Module):
    """Variational quantum circuit with mean‑field or entangled mode."""
    def __init__(self, n_qubits: int, depth: int = 1, mean_field: bool = True):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.mean_field = mean_field
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(n_qubits)) for _ in range(depth)])
        self.cnot_order = [(i, i+1) for i in range(n_qubits-1)]
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x, *params):
        for wire in range(self.n_qubits):
            qml.RX(x[wire], wires=wire)
        params = torch.cat([p.view(-1) for p in params])  # depth * n_qubits
        for i in range(self.depth):
            for j in range(self.n_qubits):
                idx = i * self.n_qubits + j
                qml.RZ(params[idx], wires=j)
            if not self.mean_field:
                for (a, b) in self.cnot_order:
                    qml.CNOT(wires=[a, b])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        out_list = []
        for i in range(batch):
            out_list.append(self.qnode(x[i], *self.params))
        return torch.stack(out_list, dim=0)

class QuantumLSTM(nn.Module):
    """LSTM cell with each gate implemented by a Pennylane variational circuit."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 depth: int = 1,
                 mean_field: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_gate = QuantumGate(n_qubits, depth, mean_field)
        self.input_gate = QuantumGate(n_qubits, depth, mean_field)
        self.update_gate = QuantumGate(n_qubits, depth, mean_field)
        self.output_gate = QuantumGate(n_qubits, depth, mean_field)
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: tuple = None):
        if states is None:
            batch_size = inputs.size(1)
            device = inputs.device
            hx = torch.zeros(batch_size, self.hidden_dim, device=device)
            cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            hx, cx = states
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between a classical nn.LSTM and QuantumLSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 depth: int = 1,
                 mean_field: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumLSTM(embedding_dim,
                                    hidden_dim,
                                    n_qubits,
                                    depth=depth,
                                    mean_field=mean_field)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QuantumLSTM", "LSTMTagger"]
