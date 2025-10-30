"""Unified estimator for classical neural networks and quantum circuits.

The module exposes a single estimator class that works with a PyTorch
neural network or a parametrized Qiskit/PyQuTiP circuit.  The public API
mirrors the two seed modules but includes shot‑noise, batched binding,
and a graph‑based fidelity graph for post‑processing.  The design is
intended for quick prototyping while keeping the implementation
read‑only and importable.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import qutip as qt

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
QuantumObservable = BaseOperator | qt.Qobj


class UnifiedQuantumEstimator:
    """Batch‑compatible estimator that supports both classical and quantum models.

    Parameters
    ----------
    model : nn.Module | QuantumCircuit | qt.Qobj
        A PyTorch neural network, a Qiskit quantum circuit, or a
        QuTiP unitary stack.  The estimator will infer the
        type at construction time and will raise if the type is not
        supported.
    shots : int | None, optional
        If provided, Gaussian shot noise is added to the deterministic
        output using a standard deviation 1/√shots.  Only applicable to
        classical models.
    seed : int | None, optional
        Random seed for the shot noise.  If ``None`` (default), ``np.random`` is used.
    """

    def __init__(
        self,
        model: Union[nn.Module, QuantumCircuit, qt.Qobj],
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed

        # Determine model type
        if isinstance(model, nn.Module):
            self._type = "torch"
        elif isinstance(model, QuantumCircuit):
            self._type = "qiskit"
        elif isinstance(model, qt.Qobj):
            self._type = "qutip"
        else:
            raise TypeError(f"Unsupported model type {type(model)!r}")

    # --------------------------------------------------------------------- #
    #  Classic (PyTorch) helpers
    # --------------------------------------------------------------------- #
    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        """Wrap 1‑D input into a batch of length 1."""
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _apply_noise(self, data: List[List[float]]) -> List[List[float]]:
        """Add Gaussian shot noise to the deterministic outputs."""
        if self.shots is None:
            return data
        rng = np.random.default_rng(self.seed)
        noisy = []
        for row in data:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # --------------------------------------------------------------------- #
    #  Quantum (Qiskit) helpers
    # --------------------------------------------------------------------- #
    def _bind_qiskit(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Return a bound circuit for a given set of parameters."""
        if len(param_values)!= len(self.model.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.model.parameters, param_values))
        return self.model.assign_parameters(mapping, inplace=False)

    # --------------------------------------------------------------------- #
    #  Quantum (QuTiP) helpers
    # --------------------------------------------------------------------- #
    def _apply_qutip_observable(
        self, state: qt.Qobj, observable: qt.Qobj
    ) -> complex:
        """Compute expectation of a QuTiP observable on a state."""
        return (state.dag() * observable * state)[0, 0]

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, QuantumObservable]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Union[float, complex]]]:
        """Compute outputs for all parameter sets.

        For a PyTorch model, ``observables`` are callables that map the
        network output tensor to a scalar.  For Qiskit circuits and
        QuTiP unitary stacks, ``observables`` are operators (``BaseOperator``
        or ``qt.Qobj``) and the function returns their expectation value
        on the resulting state.

        Parameters
        ----------
        observables : Iterable
            Sequence of observable callables or operators.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[Union[float, complex]]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[Union[float, complex]]] = []

        if self._type == "torch":
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inputs = self._ensure_batch(params)
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

            return self._apply_noise(results)

        if self._type == "qiskit":
            for params in parameter_sets:
                bound = self._bind_qiskit(params)
                state = Statevector.from_instruction(bound)
                row = [
                    state.expectation_value(obs) if isinstance(obs, BaseOperator) else 0.0
                    for obs in observables
                ]
                results.append(row)
            return results

        if self._type == "qutip":
            for params in parameter_sets:
                # For QuTiP we assume the model is a unitary stack that can be
                # applied to an initial state.  The first element of ``params``
                # is treated as a seed for a random initial state; the rest
                # are ignored in this toy implementation.
                if len(params) == 0:
                    raise ValueError("Parameter set required for QuTiP model.")
                # create a random initial state
                dim = int(np.sqrt(self.model.shape[0]))
                init = qt.rand_ket(dim)
                state = self.model * init
                row = [
                    self._apply_qutip_observable(state, obs)
                    if isinstance(obs, qt.Qobj)
                    else 0.0
                    for obs in observables
                ]
                results.append(row)
            return results

        raise RuntimeError("Unreachable code path")

    # --------------------------------------------------------------------- #
    #  Fidelity graph helper for quantum models
    # --------------------------------------------------------------------- #
    def fidelity_graph(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a weighted graph from state fidelities.

        Parameters
        ----------
        states : Sequence[qt.Qobj]
            Iterable of pure states.
        threshold : float
            Primary fidelity threshold for weight 1.0.
        secondary : float | None, optional
            Secondary threshold for weight ``secondary_weight``.
        secondary_weight : float, default 0.5
            Weight for secondary edges.

        Returns
        -------
        nx.Graph
            Weighted adjacency graph.
        """
        if self._type not in ("qiskit", "qutip"):
            raise TypeError("Fidelity graph is only defined for quantum models")

        def fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
            return abs((a.dag() * b)[0, 0]) ** 2

        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["UnifiedQuantumEstimator"]
