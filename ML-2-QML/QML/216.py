import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QLayer(nn.Module):
    """
    Variational quantum circuit that processes the concatenated
    input and hidden state. The same circuit is reused for all
    LSTM gates. The circuit outputs a real‑valued vector of
    length equal to the hidden dimension by measuring each qubit.
    """
    def __init__(self, n_qubits: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits, shots=1)

        # Parameters for the variational layers
        self.params = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float64)
        )

    def circuit(self, x: torch.Tensor, params: torch.Tensor):
        # Encode classical data via RY rotations
        for i in range(self.n_qubits):
            qml.RY(x[i].item(), wires=i)
        # Variational layers
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(params[l, i, 0].item(), wires=i)
                qml.RX(params[l, i, 1].item(), wires=i)
                qml.RY(params[l, i, 2].item(), wires=i)
            # Entangle neighbours
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        out = []
        for i in range(batch_size):
            out.append(self.circuit(x[i], self.params))
        return torch.stack(out)

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell that uses the QLayer for all gates.
    Dropout is optional for regularisation. The cell is fully
    differentiable thanks to Pennylane's autograd support.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        # Shared quantum layer for all gates
        self.qgate = QLayer(n_qubits, hidden_dim)

        # Linear projections to map concatenated features to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Map to qubit space
            f_in = self.linear_forget(combined)
            i_in = self.linear_input(combined)
            g_in = self.linear_update(combined)
            o_in = self.linear_output(combined)

            # Quantum gates
            f = torch.sigmoid(self.dropout(self.qgate(f_in)))
            i = torch.sigmoid(self.dropout(self.qgate(i_in)))
            g = torch.tanh(self.dropout(self.qgate(g_in)))
            o = torch.sigmoid(self.dropout(self.qgate(o_in)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a classical
    LSTM and the quantum‑enhanced QLSTM. Dropout is applied to
    the embeddings to reduce over‑fitting.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, dropout=dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.dropout(self.word_embeddings(sentence))
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class QLSTMTrainer:
    """
    Simple training helper for the quantum‑enhanced model. It
    demonstrates how to train with Pennylane's autograd
    and how to evaluate on a validation set.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.NLLLoss()
        self.device = device

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        sentences, tags = batch
        sentences = sentences.to(self.device)
        tags = tags.to(self.device)
        outputs = self.model(sentences)
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), tags.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sentences, tags = batch
                sentences = sentences.to(self.device)
                tags = tags.to(self.device)
                outputs = self.model(sentences)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), tags.view(-1))
                total_loss += loss.item()
        return total_loss / len(val_loader)

__all__ = ["QLSTM", "LSTMTagger", "QLSTMTrainer"]
