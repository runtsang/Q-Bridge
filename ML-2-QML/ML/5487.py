"""Hybrid module that combines classical FCL, EstimatorQNN, a classifier network, and an LSTMTagger.

Each sub‑module can be swapped with its quantum counterpart, enabling end‑to‑end differentiable training across all layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCL(nn.Module):
    """Simple fully‑connected layer with a run method."""
    def __init__(self, n_features: int = 1, hidden: int = 16):
        super().__init__()
        self.linear = nn.Linear(n_features, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        x = self.linear(thetas)
        x = torch.tanh(x)
        return self.out(x)

class EstimatorQNN(nn.Module):
    """Feed‑forward network mirroring the quantum EstimatorQNN."""
    def __init__(self, input_dim: int = 2, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

def build_classifier_circuit(num_features: int, depth: int):
    """Construct a classical classifier and its metadata."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class QLSTM(nn.Module):
    """Classical LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Tagger that uses either a classical or a quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridFCLClassifierQLSTM(nn.Module):
    """End‑to‑end hybrid model.

    The architecture is:
      * FCL  ->  EstimatorQNN  ->  Classifier  ->  LSTMTagger
    Each block can be configured independently.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.fcl = FCL(n_features=config.get("fcl_features", 1),
                       hidden=config.get("fcl_hidden", 16))
        self.estimator = EstimatorQNN(input_dim=config.get("est_input", 2),
                                      hidden=config.get("est_hidden", 8))
        self.classifier, self.enc, self.w_sizes, self.obs = build_classifier_circuit(
            num_features=config.get("clf_features", 8),
            depth=config.get("clf_depth", 2)
        )
        self.tagger = LSTMTagger(embedding_dim=config.get("emb_dim", 8),
                                 hidden_dim=config.get("lstm_hidden", 16),
                                 vocab_size=config.get("vocab_size", 1000),
                                 tagset_size=config.get("tagset_size", 10),
                                 n_qubits=config.get("n_qubits", 0))

    def forward(self, x: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input for the fully‑connected path.
        seq : torch.Tensor
            Token indices for the sequence tagger.
        """
        fcl_out = self.fcl(x)
        est_out = self.estimator(fcl_out)
        clf_out = self.classifier(est_out)
        tag_logits = self.tagger(seq)
        return torch.cat([clf_out, tag_logits], dim=-1)

__all__ = ["HybridFCLClassifierQLSTM"]
