import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Iterable, Sequence

class HybridQLSTM(nn.Module):
    """Classical LSTM with optional quantum‑style gate parametrisation."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can toggle between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = (
            HybridQLSTM(embedding_dim, hidden_dim, n_qubits)
            if n_qubits > 0
            else nn.LSTM(embedding_dim, hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.emb(sentence)
        out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.fc(out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

def build_classifier_circuit(num_features: int, depth: int):
    """Construct a classical feed‑forward classifier mirroring the quantum helper."""
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []
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

def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0
) -> np.ndarray:
    """Compute an RBF kernel matrix."""
    return np.array(
        [
            [torch.exp(-gamma * torch.sum((x - y) ** 2)).item() for y in b]
            for x in a
        ]
    )

def EstimatorQNN():
    """Return a tiny feed‑forward regression network."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()

__all__ = [
    "HybridQLSTM",
    "LSTMTagger",
    "build_classifier_circuit",
    "kernel_matrix",
    "EstimatorQNN",
]
