"""Advanced lightweight estimator utilities implemented with PyTorch modules.

This module extends the original FastBaseEstimator by adding GPU support,
batch processing, optional dropout during inference, and a convenience
method for computing gradients via torch.autograd.  The API remains
compatible with the seed implementation, but the new features allow
large‑scale evaluation on modern hardware and tighter integration
with training pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Any

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module, *, device: torch.device | str | None = None) -> None:
        """
        Parameters
        ----------
        model
            A PyTorch ``nn.Module`` that maps a batch of parameters to outputs.
        device
            Target device for evaluation.  If ``None`` the module will use
            ``torch.device('cuda' if torch.cuda.is_available() else 'cpu')``.
        """
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        dropout: bool = False,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        """
        Compute deterministic or dropout‑augmented predictions for
        multiple parameter sets.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output to a scalar.
        parameter_sets
            Sequence of 1‑D parameter lists to evaluate.
        dropout
            If ``True`` the model is run in training mode to activate
            dropout layers.
        batch_size
            Optional batch size for the evaluation loop.  When ``None`` the
            entire input set is processed in one forward pass.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        if dropout:
            self.model.train()
        else:
            self.model.eval()

        with torch.no_grad():
            # Prepare batched input
            inputs = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)

            # Process in chunks if a batch size is requested
            if batch_size is None or batch_size >= inputs.shape[0]:
                batches = [inputs]
            else:
                batches = [
                    inputs[i : i + batch_size] for i in range(0, inputs.shape[0], batch_size)
                ]

            for batch in batches:
                outputs = self.model(batch)
                for out in outputs:
                    row: List[float] = []
                    for observable in observables:
                        value = observable(out)
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    results.append(row)

        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        grad_outputs: Sequence[float] | None = None,
    ) -> List[List[float]]:
        """
        Compute gradients of observables with respect to the input parameters.

        Parameters
        ----------
        observables
            Iterable of callables that map the model output to a scalar.
        parameter_sets
            Sequence of 1‑D parameter lists to evaluate.
        grad_outputs
            Optional gradients of the output tensor to propagate back.
            If ``None`` a unit gradient is used.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[float]] = []

        self.model.eval()
        for params in parameter_sets:
            inputs = torch.as_tensor(params, dtype=torch.float32, requires_grad=True).to(
                self.device
            )
            outputs = self.model(inputs.unsqueeze(0))
            for observable in observables:
                value = observable(outputs[0])
                if grad_outputs is None:
                    grad_outputs_tensor = torch.ones_like(value)
                else:
                    grad_outputs_tensor = torch.as_tensor(
                        grad_outputs, dtype=torch.float32, device=self.device
                    )
                value.backward(grad_outputs_tensor, retain_graph=True)
                grads.append([float(inputs.grad.mean().cpu())])
                inputs.grad.zero_()
        return grads

    @staticmethod
    def make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Module:
        """
        Convenience helper that returns a simple feed‑forward network.

        Parameters
        ----------
        input_dim
            Dimensionality of the input vector.
        hidden_dims
            Sequence of hidden layer sizes.
        output_dim
            Dimensionality of the output vector.
        """
        layers: List[Any] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        dropout: bool = False,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(
            observables, parameter_sets, dropout=dropout, batch_size=batch_size
        )
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
