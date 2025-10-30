import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Callable, Sequence, List, Tuple

# --------------------------------------------------------------------------- #
# Classical LSTM with regularization
# --------------------------------------------------------------------------- #
class QLSTMClassic(nn.Module):
    """
    Drop‑in replacement of the original QLSTM that adds dropout
    to the hidden state for better generalisation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            outputs.append(self.dropout(hx.unsqueeze(0)))
        return torch.cat(outputs, dim=0), (hx, cx)

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

# --------------------------------------------------------------------------- #
# Fast estimator with optional Gaussian shot noise
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate a PyTorch model on a batch of parameter sets.
    When *shots* is given, Gaussian noise is injected to mimic
    finite‑shot measurement statistics.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().item())
                    row.append(float(val))
                if shots is not None:
                    rng = np.random.default_rng(seed)
                    noisy_row = [
                        float(rng.normal(v, max(1e-6, 1 / shots))) for v in row
                    ]
                    row = noisy_row
                results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Hybrid LSTMTagger with fast evaluation
# --------------------------------------------------------------------------- #
class QLSTMHybrid(nn.Module):
    """
    Unified interface that wraps a classical LSTMTagger and a fast estimator.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        add_shot_noise: bool = False,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMClassic(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.estimator = FastBaseEstimator(self)
        self.add_shot_noise = add_shot_noise
        self.shots = shots
        self.seed = seed

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        return self.estimator.evaluate(
            observables,
            parameter_sets,
            shots=self.shots if self.add_shot_noise else None,
            seed=self.seed,
        )

__all__ = ["QLSTMClassic", "FastBaseEstimator", "QLSTMHybrid"]
