import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QuantumGateMLP(nn.Module):
    """
    A lightweight MLP that mimics the behaviour of a quantum gate.
    It receives the concatenated input and hidden state and outputs
    a vector of size hidden_dim that will be interpreted as gate values.
    """
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int = 2, hidden_size: int = 64):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(curr_dim, hidden_size))
            layers.append(nn.ReLU())
            curr_dim = hidden_size
        layers.append(nn.Linear(curr_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QLSTM(nn.Module):
    """
    Hybrid classical‑to‑quantum LSTM cell that replaces the classical
    gate projections with a shared MLP simulating a quantum circuit.
    Dropout is added for regularisation and the module is fully
    differentiable using standard back‑propagation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Shared MLP that acts as a stand‑in for a quantum circuit
        self.gate_mlp = QuantumGateMLP(input_dim + hidden_dim, hidden_dim)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            gates = self.dropout(self.gate_mlp(combined))
            f = torch.sigmoid(gates)
            i = torch.sigmoid(gates)
            g = torch.tanh(gates)
            o = torch.sigmoid(gates)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between a pure classical LSTM
    and the hybrid QLSTM defined above. Dropout is applied to the
    embeddings to reduce over‑fitting.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, dropout: float = 0.1, use_q: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if use_q:
            self.lstm = QLSTM(embedding_dim, hidden_dim)
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
    Simple training helper that trains a given model on a toy dataset
    and reports validation loss. It demonstrates how to handle the
    hybrid LSTM in a standard PyTorch training loop.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = 'cpu'):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
