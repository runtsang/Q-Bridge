"""Quantum estimator that evaluates expectation values for a parametrized circuit with optional shot noise.

The FastBaseEstimator class accepts a Qiskit QuantumCircuit and can evaluate a list of
BaseOperator observables for many sets of parameter values.  It supports batched evaluation,
selectable simulation backends (Statevector or Aer), and optional shot noise via a QASM
simulation.  The class is fully compatible with the original API but adds richer
functionality for large‑scale quantum experiments.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel

BaseOperator = Operator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str = "statevector",
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend
        self.shots = shots
        self.seed = seed
        self.noise_model = noise_model
        self._simulator = self._build_simulator()

    def _build_simulator(self) -> AerSimulator:
        if self.backend_name == "statevector":
            return AerSimulator(method="statevector", seed_simulator=self.seed)
        elif self.backend_name == "qasm":
            return AerSimulator(method="qasm", seed_simulator=self.seed)
        else:
            raise ValueError(f"Unsupported backend '{self.backend_name}'")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {p: v for p, v in zip(self._parameters, parameter_values)}
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circuit = self._bind(values)
            if self.backend_name == "statevector":
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            elif self.backend_name == "qasm":
                job = self._simulator.run(bound_circuit, shots=self.shots)
                result = job.result()
                counts = result.get_counts(bound_circuit)
                # Convert counts to statevector approximation
                state = Statevector.from_counts(counts)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                raise RuntimeError("Unexpected backend state.")
            results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Same as evaluate but forces a QASM simulation with shot noise."""
        if self.backend_name!= "qasm":
            raise RuntimeError("evaluate_with_noise requires a 'qasm' backend.")
        return self.evaluate(observables, parameter_sets)


__all__ = ["FastBaseEstimator"]
