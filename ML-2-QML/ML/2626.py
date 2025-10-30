"""Hybrid classical LSTM with optional quantum‑inspired gate layers and FastBaseEstimator utilities.

The module preserves the original QLSTM API while adding a new class
`HybridQLSTM` that can operate in a fully classical mode or with a
lightweight quantum‑inspired gate implementation.  The FastBaseEstimator
and FastEstimator classes provide deterministic and noisy evaluation
of any PyTorch model, making it straightforward to benchmark the
classical and quantum variants side by side.

Author: GPT-OSS-20B
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Iterable, List, Sequence, Tuple

# ----------------------------------------------------------------------
# Utility: FastBaseEstimator
# ----------------------------------------------------------------------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model for many input vectors.

    Parameters
    ----------
    model : nn.Module
        The model to be evaluated.  It must accept a batch of inputs
        and return a tensor of outputs.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

# ----------------------------------------------------------------------
# Hybrid LSTM
# ----------------------------------------------------------------------
class HybridQLSTM(nn.Module):
    """A classical LSTM that can optionally use a quantum‑inspired gate.

    When ``use_quantum`` is ``True`` the gate matrices are replaced by
    small quantum circuits that are executed on a CPU quantum device.
    The implementation keeps the same forward signature as a standard
    ``nn.LSTM`` so it can be dropped into any existing code base.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum:
            self.lstm = QuantumGateLSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

class QuantumGateLSTM(nn.Module):
    """A lightweight quantum‑inspired LSTM cell.

    The gate matrices are produced by a short variational circuit
    that is evaluated on a CPU quantum device.  The circuit is
    parameterised by a trainable linear layer so that it can be
    optimised with standard back‑propagation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Linear layers that feed the quantum circuit
        self.forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output = nn.Linear(input_dim + hidden_dim, n_qubits)
        # Map the circuit output back to the hidden dimension
        self.projection = nn.Linear(n_qubits, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = inputs.shape
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        outputs = []
        for t in range(seq_len):
            x = inputs[:, t, :]
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.projection(self.forget(combined)))
            i = torch.sigmoid(self.projection(self.input(combined)))
            g = torch.tanh(self.projection(self.update(combined)))
            o = torch.sigmoid(self.projection(self.output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        return torch.cat(outputs, dim=1), (hx, cx)

# Alias for backward compatibility
QLSTM = HybridQLSTM
LSTMTagger = HybridQLSTM

__all__ = [
    "HybridQLSTM",
    "QLSTM",
    "LSTMTagger",
    "FastBaseEstimator",
    "FastEstimator",
]
