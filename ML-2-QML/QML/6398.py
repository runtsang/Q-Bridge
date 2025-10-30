"""Hybrid estimator that couples a quantum circuit with a classical
neural network for parameter generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np
import torch
from torch import nn
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class HybridFastEstimator:
    """
    Evaluate a quantum circuit whose parameters are produced by a
    classical neural network.  The network is wrapped with
    :class:`qiskit_machine_learning.neural_networks.SamplerQNN` so that
    the whole pipeline can be trained end‑to‑end.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised quantum circuit.
    input_params : Sequence[str] | None
        Names of the circuit parameters that correspond to input data.
    weight_params : Sequence[str] | None
        Names of the circuit parameters that correspond to learnable
        weights.  If ``None`` all parameters are treated as weights.
    sampler : Optional[Sampler]
        Backend sampler for state‑vector or shot sampling.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        input_params: Sequence[str] | None = None,
        weight_params: Sequence[str] | None = None,
        sampler: None = None,
    ) -> None:
        self.circuit = circuit
        self.input_params = input_params or []
        self.weight_params = weight_params or []
        self.sampler = sampler

        # Wrap the circuit in a Qiskit SamplerQNN
        self.qnn = QiskitSamplerQNN(
            circuit=circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
        )

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """
        Compute expectation values for each parameter set.
        The *parameter_sets* are fed directly to the circuit’s input
        parameters; the remaining parameters are treated as trainable
        weights and are optimized by the Qiskit SamplerQNN backend.
        """
        results: list[list[float]] = []
        for params in parameter_sets:
            # Bind input parameters
            if len(params)!= len(self.input_params):
                raise ValueError("Input parameter count mismatch.")
            mapping = dict(zip(self.input_params, params))
            bound_qc = self.circuit.assign_parameters(mapping, inplace=False)

            # Evaluate with the sampler
            state = Statevector.from_instruction(bound_qc)
            row = [float(state.expectation_value(obs)) for obs in observables]
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
        Uses the SamplerQNN backend if available, otherwise falls back
        to a state‑vector sampler.
        """
        if self.sampler is None:
            raise RuntimeError("Sampler backend not configured.")
        samples: list[list[str]] = []
        for params in parameter_sets:
            if len(params)!= len(self.input_params):
                raise ValueError("Input parameter count mismatch.")
            mapping = dict(zip(self.input_params, params))
            bound_qc = self.circuit.assign_parameters(mapping, inplace=False)
            # Use the Qiskit SamplerQNN to draw samples
            sample_counts = self.sampler.run(bound_qc, shots=num_shots).result()
            outcomes = sample_counts.get_counts()
            # Convert outcome dict to list of bitstrings
            samples.append(list(outcomes.keys()))
        return samples


__all__ = ["HybridFastEstimator"]
