"""Enhanced quantum estimator with batched simulation, parameterized observables, and shot noise."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

try:
    from qiskit.providers.aer import AerSimulator
    _AER_AVAILABLE = True
except ImportError:
    _AER_AVAILABLE = False
    AerSimulator = None

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit with optional shot noise."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: Optional[AerSimulator] = None,
        shots: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.backend = backend or (AerSimulator() if _AER_AVAILABLE else None)
        self.shots = shots
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[float]] = []
        for values in parameter_sets:
            bound_circ = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [float(state.expectation_value(obs).real) for obs in observables]
            else:
                if self.backend is None:
                    raise RuntimeError("Aer backend is required for shot simulation.")
                job = self.backend.run(bound_circ, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                probs = {k: v / self.shots for k, v in counts.items()}
                samples = []
                for bitstring, p in probs.items():
                    samples.extend([bitstring] * int(p * self.shots))
                while len(samples) < self.shots:
                    samples.append(np.random.choice(list(probs.keys()), p=list(probs.values())))
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring in samples:
                        exp += float(obs.expectation_value(Statevector.from_label(bitstring)).real)
                    row.append(exp / self.shots)
            results.append(row)
        return results

__all__ = ["FastBaseEstimator"]
