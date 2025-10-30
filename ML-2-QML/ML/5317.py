import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import itertools
from typing import List, Tuple, Iterable, Sequence, Callable, Optional

# ---------- Classical Graph Neural Network Utilities ----------

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    activations = []
    for features, _ in samples:
        layer_acts = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_acts.append(current)
        activations.append(layer_acts)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5):
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ---------- Estimator Primitives ----------

class FastBaseEstimator:
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]], parameter_sets: Sequence[Sequence[float]], *, shots: int | None = None, seed: int | None = None):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1/shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# ---------- Hybrid Sequence Models ----------

class QLSTM(nn.Module):
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

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# ---------- Transformer‑Based Classifier ----------

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int, ffn_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor):
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# ---------- Hybrid Graph Neural Network Class ----------

class GraphQNN:
    """Hybrid graph neural network with classical or quantum propagation and optional sequence models."""

    def __init__(self, qnn_arch: Sequence[int], use_qubits: bool = False, device: str = 'cpu'):
        self.arch = list(qnn_arch)
        self.use_qubits = use_qubits
        self.device = device
        self.weights: Optional[List[torch.Tensor]] = None
        self.training_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.target_weight: Optional[torch.Tensor] = None
        self.lstm_tagger: Optional[LSTMTagger] = None
        self.text_classifier: Optional[TextClassifier] = None

    def random_network(self, samples: int):
        if not self.use_qubits:
            self.arch, self.weights, self.training_data, self.target_weight = random_network(self.arch, samples)
        else:
            raise NotImplementedError("Quantum network generation is implemented in the quantum module.")
        return self.arch, self.weights, self.training_data, self.target_weight

    def feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        if not self.use_qubits:
            return feedforward(self.arch, self.weights, samples)
        else:
            raise NotImplementedError("Quantum feedforward is implemented in the quantum module.")

    def fidelity_adjacency(self, states: Sequence[torch.Tensor], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5):
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def build_lstm_tagger(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        self.lstm_tagger = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits)

    def build_text_classifier(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int, ffn_dim: int, num_classes: int, dropout: float = 0.1):
        self.text_classifier = TextClassifier(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim, num_classes, dropout)

    def forward(self, inputs: torch.Tensor):
        if self.target_weight is None:
            raise RuntimeError("GraphQNN not initialized with weights. Call random_network first.")
        return self.target_weight @ inputs.t()

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]], parameter_sets: Sequence[Sequence[float]]):
        estimator = FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets)

__all__ = [
    "GraphQNN",
    "QLSTM",
    "LSTMTagger",
    "TextClassifier",
    "PositionalEncoder",
    "TransformerBlock",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FastBaseEstimator",
    "FastEstimator",
]
