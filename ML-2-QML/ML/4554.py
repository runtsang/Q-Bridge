import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

class QLSTM(nn.Module):
    """Classical LSTM cell with linear gates, mirroring the quantum interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0) -> None:
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

class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridClassifier(nn.Module):
    """Hybrid classical classifier that can optionally use LSTM or CNN feature extractors."""
    def __init__(self, num_features: int, depth: int = 3, use_lstm: bool = False, use_cnn: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # Feature extractor
        if self.use_lstm:
            self.feature_extractor = LSTMTagger(embedding_dim=4, hidden_dim=16, vocab_size=1000, tagset_size=2, n_qubits=4)
        elif self.use_cnn:
            self.feature_extractor = QCNNModel()
        else:
            self.feature_extractor = None

        # Classifier head
        layers = []
        in_dim = 2 if self.use_lstm else (4 if self.use_cnn else num_features)
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor:
            x = self.feature_extractor(x)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

def build_classifier_circuit(num_features: int, depth: int, use_lstm: bool = False, use_cnn: bool = False) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Construct a hybrid classifier model and expose metadata."""
    if use_lstm:
        model = HybridClassifier(num_features, depth, use_lstm=True)
    elif use_cnn:
        model = HybridClassifier(num_features, depth, use_cnn=True)
    else:
        model = HybridClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(2))
    return model, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
