"""Hybrid classical LSTM with optional kernel gating and batch evaluation.

The implementation extends the original QLSTM by adding:
* kernel‑based forget‑gate modulation (RBF kernel).
* an :class:`FastBaseEstimator` wrapper for fast batch evaluation
  and optional shot‑noise simulation.
The class signature is identical to the original QLSTM, so it can be
used as a drop‑in replacement for sequence tagging tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from FastBaseEstimator import FastBaseEstimator


class HybridQLSTM(nn.Module):
    """Hybrid LSTM with classical gates and optional kernel modulation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        kernel_gate: bool = False,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.kernel_gate = kernel_gate

        # Classical linear gates
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        # Kernel parameters
        if self.kernel_gate:
            self.gamma = gamma
            self.kernel_param = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def _kernel_score(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """RBF kernel between input and hidden state."""
        diff = x - h
        return torch.exp(-self.kernel_param * torch.sum(diff * diff, dim=-1, keepdim=True))

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
            if self.kernel_gate:
                f = f * self._kernel_score(x, hx)
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def evaluate(
        self,
        observables: list[callable],
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        """
        Batch‑evaluate scalar observables over a list of *parameter_sets*.
        Each parameter set is interpreted as a candidate hidden state; the
        model is run in evaluation mode.  Optional Gaussian shot noise
        can be added via *shots*.
        """
        estimator = FastBaseEstimator(self)
        raw = estimator.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy
