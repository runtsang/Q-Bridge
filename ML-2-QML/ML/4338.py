"""Hybrid estimator combining classical primitives with optional quantum support.

The module defines a FastBaseEstimator that can evaluate any PyTorch model
and optionally add Gaussian shot noise.  It also ships with convenient
primitives for convolution, LSTM and self‑attention that can be composed
into a single model.  The design follows a *combination* scaling paradigm:
classical and quantum components are treated as independent building blocks
that can be mixed at the model level.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ConvFilter(nn.Module):
    """Classical 2‑D convolutional filter with thresholding.

    The filter is expressed as a single‑channel 2‑D convolution followed
    by a sigmoid activation.  The threshold parameter allows the user
    to bias the sigmoid curve.  The module is intentionally lightweight
    so that it can be dropped into a larger network without extra
    overhead.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, height, width) or (height, width)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(1)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])


class QLSTM(nn.Module):
    """Hybrid LSTM that can switch between classical and quantum gates.

    When ``n_qubits > 0`` the gates are implemented by a small
    parameter‑ised quantum circuit using a toy “quantum” layer.  For the
    classical case the gates reduce to the standard linear layers.
    """

    class _QuantumGate(nn.Module):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = nn.ModuleList(
                [nn.Linear(1, 1, bias=False) for _ in range(n_wires)]
            )
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(1)) for _ in range(n_wires)]
            )
            self.measure = nn.Linear(n_wires, 1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.cat([gate(x.unsqueeze(-1)) for gate in self.encoder], dim=-1)
            out = out + torch.sin(self.params[0])  # mock quantum effect
            return torch.sigmoid(self.measure(out))

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        gate_dim = hidden_dim
        if n_qubits > 0:
            self.forget = self._QuantumGate(n_qubits)
            self.input_gate = self._QuantumGate(n_qubits)
            self.update = self._QuantumGate(n_qubits)
            self.output = self._QuantumGate(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            self.forget = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.input_gate = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.update = nn.Linear(input_dim + hidden_dim, gate_dim)
            self.output = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input_gate(combined))
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
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class ClassicalSelfAttention(nn.Module):
    """Scaled dot‑product self‑attention implemented in PyTorch."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.matmul(scores, value)


class FastBaseEstimator:
    """Hybrid estimator that evaluates a PyTorch model and optionally adds shot noise."""

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
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

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
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "ConvFilter",
    "QLSTM",
    "ClassicalSelfAttention",
    "FastBaseEstimator",
    "FastEstimator",
]
