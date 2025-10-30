import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

class DepthSepResCNN(nn.Module):
    """Depth‑wise separable CNN with residual connections."""
    def __init__(self, in_channels: int = 1, base: int = 8, depth: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            in_c = base * (2 ** i)
            conv = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, groups=in_c)
            depthwise = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
            block = nn.Sequential(conv, nn.ReLU(), depthwise, nn.ReLU())
            if i > 0:
                block.add_module("res", nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0))
            self.blocks.append(block)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
            x = self.pool(x)
        return x

class ClassicalFCLayer(nn.Module):
    """Classical surrogate for a quantum fully‑connected layer."""
    def __init__(self, in_features: int, out_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        theta = torch.as_tensor(list(thetas), dtype=torch.float32)
        return self.net(theta).mean().unsqueeze(0)

def feedforward_graph(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                       samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    activations = []
    for feat, _ in samples:
        layer = [feat]
        current = feat
        for w in weights:
            current = torch.tanh(w @ current)
            layer.append(current)
        activations.append(layer)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm) ** 2)

def fidelity_graph(states: List[torch.Tensor], threshold: float,
                   *, secondary: float | None = None, secondary_weight: float = 0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue
            fid = state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Tagger that can use either classical or quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, use_quantum: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum:
            from.qlstm import QLSTM
            self.lstm = QLSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridNAT(nn.Module):
    """Hybrid architecture combining classical CNN, FC, graph, and LSTM."""
    def __init__(self, cnn_channels: int = 1, base: int = 8, depth: int = 3,
                 fc_out: int = 4, lstm_dim: int = 32, vocab_size: int = 1000,
                 tagset_size: int = 10):
        super().__init__()
        self.cnn = DepthSepResCNN(cnn_channels, base, depth)
        self.fc = ClassicalFCLayer(16 * (2 ** (depth - 1)), fc_out)
        self.lstm = LSTMTagger(embedding_dim=fc_out, hidden_dim=lstm_dim,
                               vocab_size=vocab_size, tagset_size=tagset_size,
                               use_quantum=False)

    def forward(self, x: torch.Tensor, sentences: torch.Tensor) -> Tuple[torch.Tensor, nx.Graph]:
        features = self.cnn(x)
        flat = features.view(features.size(0), -1)
        out = self.fc(flat)
        graph = fidelity_graph([torch.tensor(row) for row in out.tolist()], threshold=0.8)
        lstm_out, _ = self.lstm.lstm(out.unsqueeze(0))
        return lstm_out, graph

__all__ = ['HybridNAT', 'DepthSepResCNN', 'ClassicalFCLayer',
           'feedforward_graph','state_fidelity', 'fidelity_graph',
           'ClassicalQLSTM', 'LSTMTagger']
