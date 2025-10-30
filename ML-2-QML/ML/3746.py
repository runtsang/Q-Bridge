"""Hybrid classical estimator that supports PyTorch modules and QCNNModel
with optional Gaussian shot noise."""
from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import torch
from torch import nn

# Import the original classical base classes
from.FastBaseEstimator import FastBaseEstimator, FastEstimator
# Import the QCNN model factory
from.QCNN import QCNN, QCNNModel

# Type alias for an observable that maps a network output to a scalar
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D parameter list into a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """
    Wrapper that can evaluate either a :class:`torch.nn.Module`
    (including :class:`QCNNModel`) or any custom PyTorch model.

    Parameters
    ----------
    model : Union[nn.Module, QCNNModel]
        The model to evaluate.

    Notes
    -----
    * When *shots* is ``None`` the evaluation is deterministic.
    * When *shots* is provided, Gaussian noise with variance
      ``1 / shots`` is added to each output, mimicking shot noise.
    * The default observable returns the mean of the last dimension.
    """

    def __init__(self, model: Union[nn.Module, QCNNModel]) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or QCNNModel")
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on a list of parameter sets.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Functions that map the network output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of 1‑D parameter lists.
        shots : int, optional
            If provided, Gaussian noise with variance ``1/shots`` is added.
        seed : int, optional
            Seed for the noise generator.

        Returns
        -------
        List[List[float]]
            A matrix of shape ``(len(parameter_sets), len(observables))``.
        """
        # Default observable if none provided
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]

        # Choose the base estimator
        base_cls = FastEstimator if shots is not None else FastBaseEstimator
        base_est = base_cls(self.model)

        # Delegate to the base estimator
        raw_results = base_est.evaluate(observables, parameter_sets)

        # If shots were requested, the FastEstimator already added noise.
        return raw_results

    @staticmethod
    def create_qcnn() -> QCNNModel:
        """Convenience constructor for the QCNNModel."""
        return QCNN()
