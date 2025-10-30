import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Iterable, List

class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell that supports either classical linear gates or quantum gates.

    The `use_quantum` flag activates the quantum implementation supplied by a
    :class:`qml.QLayer`.  When `n_qubits` is zero the cell behaves like a
    conventional PyTorch LSTM cell, enabling drop‑in replacements for existing
    codebases.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = n_qubits > 0

        gate_dim = hidden_dim
        if self.use_quantum:
            # Quantum gates output a vector of size `n_qubits`; a linear layer
            # projects the concatenated input into this space.
            self.forget = nn.ModuleList([nn.Linear(input_dim + hidden_dim, n_qubits) for _ in range(4)])
        else:
            self.forget = nn.ModuleList([nn.Linear(input_dim + hidden_dim, gate_dim) for _ in range(4)])

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget[0](combined))
            i = torch.sigmoid(self.forget[1](combined))
            g = torch.tanh(self.forget[2](combined))
            o = torch.sigmoid(self.forget[3](combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class FastEstimator:
    """Evaluator that runs a model over a batch of parameter sets and returns observables.

    The `observables` are callables that transform the model output into a scalar.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
                 parameter_sets: Iterable[Iterable[float]]) -> List[List[float]]:
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = [float(obs(outputs)) for obs in observables]
                results.append(row)
        return results

class FastEstimatorWithNoise(FastEstimator):
    """Adds Gaussian shot noise to deterministic outputs."""
    def __init__(self, model: nn.Module, shots: Optional[int] = None, seed: Optional[int] = None):
        super().__init__(model)
        self.shots = shots
        self.rng = None
        if shots is not None:
            import numpy as np
            self.rng = np.random.default_rng(seed)

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
                 parameter_sets: Iterable[Iterable[float]]) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        noisy = []
        for row in raw:
            noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridQLSTM", "LSTMTagger", "FastEstimator", "FastEstimatorWithNoise"]
