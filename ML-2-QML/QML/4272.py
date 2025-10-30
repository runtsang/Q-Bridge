"""Quantum‑centric hybrid estimator that can also evaluate a classical neural network.

The class accepts a parameterised quantum circuit, and optionally a PyTorch
model and autoencoder.  The `evaluate` method returns expectation values of
the supplied observables; if a classical model is present, its outputs are
also returned for each parameter set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastHybridEstimator:
    """
    Hybrid estimator that performs quantum expectation‑value sampling and
    optional classical neural‑network inference.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterised circuit to evaluate.
    classical_model : nn.Module, optional
        PyTorch model that will be evaluated on the same parameter vector.
    autoencoder : nn.Module, optional
        Autoencoder that preprocesses the parameter vector before the
        classical model.
    shots : int, optional
        Number of shots for the QASM simulator.  If ``None`` a state‑vector
        evaluation is used.
    seed : int, optional
        Random seed for shot sampling.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        classical_model: nn.Module | None = None,
        autoencoder: nn.Module | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.classical_model = classical_model
        self.autoencoder = autoencoder
        self.shots = shots
        self.seed = seed
        self.parameters = list(circuit.parameters)

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(self.parameters):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self.parameters, params))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _preprocess(self, params: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is None:
            return params
        with torch.no_grad():
            return self.autoencoder.encode(params)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute quantum expectation values for each parameter set.
        """
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound = self._bind(params)
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    bound,
                    backend=Aer.get_backend("qasm_simulator"),
                    shots=self.shots,
                )
                counts = job.result().get_counts(bound)
                probs = np.array(list(counts.values())) / self.shots
                states = np.array(list(counts.keys()), dtype=int)
                row = [float(np.sum(states * probs)) for _ in observables]
            results.append(row)
        return results

    def evaluate_classical(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Forward the same parameters through the optional classical model.
        """
        if self.classical_model is None:
            raise RuntimeError("No classical model supplied.")
        self.classical_model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                processed = self._preprocess(batch)
                outputs = self.classical_model(processed)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_combined(
        self,
        quantum_observables: Iterable[BaseOperator],
        classical_observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> dict:
        """
        Return a dictionary with keys ``quantum`` and ``classical``.
        """
        quantum = self.evaluate(quantum_observables, parameter_sets)
        classical = (
            self.evaluate_classical(classical_observables, parameter_sets)
            if self.classical_model
            else []
        )
        return {"quantum": quantum, "classical": classical}

# Example helper: a quantum fully‑connected layer circuit
def FCL(n_qubits: int = 1) -> QuantumCircuit:
    """Return a simple parameterised circuit mimicking a fully‑connected layer."""
    qc = QuantumCircuit(n_qubits)
    theta = qc._parameter("theta")
    qc.h(range(n_qubits))
    qc.ry(theta, range(n_qubits))
    qc.measure_all()
    return qc

__all__ = ["FastHybridEstimator", "FCL"]
