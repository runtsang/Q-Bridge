"""Hybrid LSTM with optional classical and quantum‑inspired components."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Estimator utilities
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate a PyTorch model for a list of parameter sets."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self, observables: list[callable], parameter_sets: list[list[float]]) -> list[list[float]]:
        observables = observables or [lambda x: x.mean(dim=-1)]
        results: list[list[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(inp)
                row: list[float] = []
                for obs in observables:
                    val = obs(out)
                    row.append(float(val.mean().item()))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: list[callable],
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: list[list[float]] = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

# ----------------------------------------------------------------------
# Classical LSTM cell
# ----------------------------------------------------------------------
class ClassicalQLSTM(nn.Module):
    """Standard LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    @staticmethod
    def _init_states(
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, inputs.size(-1), device=device)
        cx = torch.zeros(batch_size, inputs.size(-1), device=device)
        return hx, cx

# ----------------------------------------------------------------------
# Hybrid classification head (quantum‑inspired)
# ----------------------------------------------------------------------
class QuantumInspiredHead(nn.Module):
    """A lightweight head that emulates a quantum expectation value."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return torch.sigmoid(logits + self.shift)

# ----------------------------------------------------------------------
# Main hybrid LSTM tagger
# ----------------------------------------------------------------------
class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM for sequence tagging that can operate purely classically
    or delegate the gating operations to a quantum module supplied by the
    user.  The class preserves the public API of the original QLSTM
    module while providing additional utilities for batch evaluation
    and a quantum‑inspired classification head.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        quantum_gate: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Gate module selection
        if n_qubits > 0 or quantum_gate is not None:
            self.lstm_gate = quantum_gate or ClassicalQLSTM(embedding_dim, hidden_dim)
        else:
            self.lstm_gate = ClassicalQLSTM(embedding_dim, hidden_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Quantum‑inspired head for optional binary classification
        self.quantum_head = QuantumInspiredHead(hidden_dim, shift=0.0)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.

        Parameters
        ----------
        sentence
            Tensor of token indices of shape ``(batch, seq_len)``.
        """
        embeds = self.word_embeddings(sentence).transpose(0, 1)  # (seq_len, batch, embed)
        gate_out, _ = self.lstm_gate(embeds)
        lstm_out, _ = self.lstm(gate_out.transpose(0, 1))
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

    # ------------------------------------------------------------------
    # Convenience evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: list[callable],
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        """Delegate to :class:`FastEstimator` for batch evaluation."""
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

# ----------------------------------------------------------------------
# Aliases to preserve backward compatibility with the original QLSTM module
# ----------------------------------------------------------------------
QLSTM = HybridQLSTM
LSTMTagger = HybridQLSTM

__all__ = ["HybridQLSTM", "QLSTM", "LSTMTagger", "FastEstimator", "FastBaseEstimator"]
