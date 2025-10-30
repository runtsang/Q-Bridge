"""
FastBaseEstimator: hybrid classical‑plus‑quantum estimator with batched inference.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Tuple

# --------------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape (1, len(values)) for a single parameter set."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# --------------------------------------------------------------------------- #
#  Core estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Hybrid estimator that evaluates a PyTorch neural network and optionally a
    variational quantum circuit.

    Parameters
    ----------
    model : nn.Module
        Classical model that maps input parameters to a feature vector.
    quantum_evaluator : Callable[[Sequence[Sequence[float]]], List[List[complex]]], optional
        Callable that takes a list of parameter sets and returns a list of
        complex expectation values.  The callable must be
        differentiable if back‑propagation through the quantum layer is
        required.  If ``None`` the estimator will only evaluate the
        classical model.
    device : str | torch.device, optional
        Torch device on which the model will run.
    """

    def __init__(
        self,
        model: nn.Module,
        quantum_evaluator: Callable[[Sequence[Sequence[float]]], List[List[complex]]] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.quantum_evaluator = quantum_evaluator
        self.device = torch.device(device)

    # --------------------------------------------------------------------- #
    #  Classical evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate the classical model for a batch of input parameters.

        Parameters
        ----------
        observables : Iterable[Callable]
            Iterable of scalar observables that map the model output to a
            scalar.  If the iterable is empty a default mean observable is
            used.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[float]]
            Nested list of scalar observables for each parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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

    # --------------------------------------------------------------------- #
    #  Quantum evaluation
    # --------------------------------------------------------------------- #
    def evaluate_quantum(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Forward a batch of parameter sets through the quantum evaluator.

        Parameters
        ----------
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.  The dimensionality must match the
            quantum evaluator's expectations.
        shots : int | None, optional
            Number of shots for stochastic evaluation.  If ``None`` the
            evaluator is expected to return exact expectation values.
        seed : int | None, optional
            Random seed for shot sampling.

        Returns
        -------
        List[List[complex]]
            Nested list of complex expectation values.
        """
        if self.quantum_evaluator is None:
            raise RuntimeError("No quantum evaluator provided.")
        return self.quantum_evaluator(parameter_sets, shots=shots, seed=seed)

    # --------------------------------------------------------------------- #
    #  Simple training loop
    # --------------------------------------------------------------------- #
    def train(
        self,
        train_data: Sequence[Tuple[Sequence[float], List[complex]]],
        epochs: int = 100,
        lr: float = 1e-3,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Train the classical model only.  The quantum evaluator is treated as a
        black box; its parameters are not updated.

        Parameters
        ----------
        train_data : Sequence[Tuple[Sequence[float], List[complex]]]
            Each element is a tuple ``(params, target_expectations)``.  The
            target expectations are complex numbers for each observable.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate for the Adam optimiser.
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
            Custom loss function that accepts predictions and targets.  If
            ``None`` the mean squared error over real and imaginary parts is
            used.
        verbose : bool
            If ``True`` print progress each epoch.
        """
        if loss_fn is None:
            loss_fn = lambda pred, tgt: ((pred.real - tgt.real) ** 2 + (pred.imag - tgt.imag) ** 2).mean()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for params, targets in train_data:
                optimizer.zero_grad()
                # Forward
                features = self.model(torch.tensor(params, dtype=torch.float32, device=self.device))
                # Quantum evaluation (deterministic)
                preds = self.quantum_evaluator([features.cpu().numpy().tolist()], shots=None)[0]
                preds_tensor = torch.tensor(preds, dtype=torch.complex64, device=self.device)
                tgt_tensor = torch.tensor(targets, dtype=torch.complex64, device=self.device)
                loss = loss_fn(preds_tensor, tgt_tensor)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:04d}/{epochs:04d}  loss={epoch_loss / len(train_data):.6f}")

# --------------------------------------------------------------------------- #
#  Expose public API
# --------------------------------------------------------------------------- #
__all__ = ["FastBaseEstimator"]
