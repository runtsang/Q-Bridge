"""Quantum‑enhanced estimator with variational training and shot‑noise support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import pennylane as qml
import numpy as np
import torch
from torch import optim

# Type alias for observables compatible with Pennylane
PennylaneObservable = qml.operation.Operation


class FastBaseEstimator:
    """Evaluate and train a variational quantum circuit.

    Parameters
    ----------
    circuit : Callable[[torch.Tensor], None]
        User‑defined circuit function that accepts a 1‑D torch tensor of parameters
        and applies gates to the Pennylane device.
    observables : Sequence[PennylaneObservable], optional
        Observables whose expectation values will be returned.
    device : str, optional
        Pennylane device name. Defaults to ``'default.qubit'``.
    shots : int | None, optional
        Number of shots for simulation. ``None`` uses the device's default.
    seed : int | None, optional
        Random seed for the device.
    """

    def __init__(
        self,
        circuit: Callable[[torch.Tensor], None],
        observables: Sequence[PennylaneObservable] | None = None,
        device: str = "default.qubit",
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._circuit_fn = circuit
        self._observables = list(observables) if observables is not None else []
        self._shots = shots
        self._seed = seed

        dev_kwargs = {"shots": shots} if shots is not None else {}
        if seed is not None:
            dev_kwargs["seed"] = seed
        self._device = qml.device(device, **dev_kwargs)

        # Build a QNode that returns expectation values for the supplied observables
        def _qnode(params: torch.Tensor):
            self._circuit_fn(params)
            return [qml.expval(obs) for obs in self._observables]

        self._qnode = qml.QNode(_qnode, self._device, interface="torch")

    def evaluate(
        self,
        observables: Iterable[PennylaneObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[PennylaneObservable], optional
            Observables to evaluate. If ``None``, uses the observables supplied at init.
        parameter_sets : Sequence[Sequence[float]], optional
            Iterable of parameter sequences. If ``None``, returns an empty list.

        Returns
        -------
        List[List[complex]]
            Matrix of shape ``(n_samples, n_observables)``.
        """
        if observables is None:
            observables = self._observables
        else:
            observables = list(observables)

        if parameter_sets is None:
            return []

        results: List[List[complex]] = []
        for params in parameter_sets:
            param_tensor = torch.as_tensor(params, dtype=torch.float32)
            expvals = self._qnode(param_tensor)
            results.append([float(e) for e in expvals])
        return results

    def fit(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer_cls: Callable[[List[torch.nn.Parameter]], optim.Optimizer] | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[float]:
        """Train the circuit parameters to match target expectation values.

        Parameters
        ----------
        inputs : Sequence[Sequence[float]]
            Training parameter sets.
        targets : Sequence[Sequence[float]]
            Desired expectation values for each observable.
        loss_fn : callable, optional
            Loss function accepting ``(pred, target)``. Defaults to MSE.
        optimizer_cls : callable, optional
            Optimizer constructor accepting a list of parameters. Defaults to Adam.
        epochs : int, default 10
            Number of training epochs.
        batch_size : int, default 32
            Mini‑batch size.
        shots : int | None, optional
            Override the number of shots for each evaluation.
        seed : int | None, optional
            Override the random seed for the device.

        Returns
        -------
        List[float]
            Training loss history.
        """
        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()
        if optimizer_cls is None:
            optimizer_cls = optim.Adam

        if shots is not None:
            # Re‑create the device and QNode with the new shot count
            dev_kwargs = {"shots": shots}
            if seed is not None:
                dev_kwargs["seed"] = seed
            self._device = qml.device(self._device.name, **dev_kwargs)

            def _qnode(params: torch.Tensor):
                self._circuit_fn(params)
                return [qml.expval(obs) for obs in self._observables]

            self._qnode = qml.QNode(_qnode, self._device, interface="torch")

        # Prepare data
        X = torch.as_tensor(inputs, dtype=torch.float32)
        Y = torch.as_tensor(targets, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Parameters are the circuit's trainable weights; they are accessible via the QNode's parameters
        params = self._qnode.trainable_params
        optimizer = optimizer_cls(params)

        history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = torch.stack([self._qnode(p) for p in batch_x])
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
        return history


__all__ = ["FastBaseEstimator"]
