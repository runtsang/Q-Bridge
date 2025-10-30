"""Hybrid estimator that bridges a PyTorch model with a quantum circuit.

The class can evaluate deterministic classical outputs, or, if a quantum
circuit is supplied, it forwards the model’s output to the circuit and
computes expectation values on a state‑vector simulator.
Shot noise can be added for emulating finite‑shot experiments.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np
import torch
from torch import nn
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """
    Evaluate a PyTorch model and an optional parametrised quantum circuit
    for a collection of parameter sets.

    Parameters
    ----------
    model : nn.Module
        Classical neural network that maps input parameters to output
        features which may be used as quantum circuit parameters.
    circuit : QuantumCircuit, optional
        Parametrised quantum circuit.  If supplied, the output of *model*
        is interpreted as the circuit’s parameters.
    shots : int, optional
        Number of shots for stochastic evaluation.  If ``None`` the
        simulation is deterministic.
    seed : int, optional
        Random seed for shot noise.
    """

    def __init__(
        self,
        model: nn.Module,
        circuit: QuantumCircuit | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _bind_circuit(self, param_vals: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with parameters bound to *param_vals*."""
        if self.circuit is None:
            raise RuntimeError("Quantum circuit not configured.")
        if len(param_vals)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, param_vals))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """
        Compute expectation values for each parameter set.

        For classical models the *observables* are callables that operate
        on the network output.  For quantum circuits they are
        :class:`~qiskit.quantum_info.operators.base_operator.BaseOperator`
        instances.  A mixture is not supported.
        """

        results: list[list[float]] = []

        for params in parameter_sets:
            # Forward through the classical network
            inputs = _ensure_batch(params)
            outputs = self.model(inputs)

            # If a quantum circuit is present, use the network output as
            # circuit parameters
            if self.circuit is not None:
                if outputs.shape[-1]!= len(self.circuit.parameters):
                    raise ValueError(
                        "Network output size does not match circuit parameter count."
                    )
                qc = self._bind_circuit(outputs.squeeze().tolist())
                state = Statevector.from_instruction(qc)
                row = [float(state.expectation_value(obs)) for obs in observables]  # type: ignore[arg-type]
            else:
                # Classical evaluation
                row = []
                for obs in observables:  # type: ignore[arg-type]
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

            # Add shot noise if requested
            if self.shots is not None:
                noise_std = max(1e-6, 1 / np.sqrt(self.shots))
                row = [float(v + self._rng.normal(0, noise_std)) for v in row]

            results.append(row)

        return results

    def sample(
        self,
        measurement: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        *,
        num_shots: int,
    ) -> list[list[str]]:
        """
        Draw samples from the quantum circuit for each parameter set.

        Parameters
        ----------
        measurement : BaseOperator
            Observable to measure.  Typically a Pauli operator.
        parameter_sets : Sequence[Sequence[float]]
            Classical inputs fed to the network.
        num_shots : int
            Number of shots per parameter set.

        Returns
        -------
        samples : list[list[str]]
            A list of measurement outcomes for each parameter set.
        """
        if self.circuit is None:
            raise RuntimeError("Quantum circuit not configured for sampling.")
        samples: list[list[str]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            outputs = self.model(inputs)
            if outputs.shape[-1]!= len(self.circuit.parameters):
                raise ValueError(
                    "Network output size does not match circuit parameter count."
                )
            qc = self._bind_circuit(outputs.squeeze().tolist())
            state = Statevector.from_instruction(qc)
            probs = state.probabilities_dict()
            outcome = np.random.choice(
                list(probs.keys()), size=num_shots, p=list(probs.values())
            )
            samples.append(outcome.tolist())
        return samples


__all__ = ["HybridFastEstimator"]
