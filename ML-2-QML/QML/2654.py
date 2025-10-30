"""Hybrid quantum estimator that can evaluate a parameterised circuit or a hybrid circuit with classical encoding.

The estimator can compute expectation values of observables and optionally simulate shot noise
using the Aer simulator.  It also provides a helper to construct a quantum circuit that
encodes classical data via a simple rotation and then applies a quantum layer inspired by Quantum‑NAT.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import RX, RY, RZ, CX, H, SX


class HybridCircuit(QuantumCircuit):
    """Parameterised circuit that encodes data and applies a quantum layer."""

    def __init__(self, n_wires: int = 4):
        super().__init__(n_wires)
        # Data encoding: simple Ry on each wire
        for i in range(n_wires):
            self.append(RY(Parameter(f"theta_{i}")), [i])
        # Quantum layer inspired by Quantum‑NAT
        self.append(CX, [0, 1])
        self.append(RX(Parameter("phi_0")), [0])
        self.append(RY(Parameter("phi_1")), [1])
        self.append(RZ(Parameter("phi_2")), [2])
        self.append(H, [3])
        self.append(SX, [2])
        self.append(CX, [3, 0])


class FastHybridEstimator:
    """Estimator for quantum circuits with optional shot noise.

    The estimator accepts a QuantumCircuit (or a subclass) and a list of
    BaseOperator observables.  It can evaluate expectation values using a
    statevector simulator or an Aer simulator with a specified number of shots.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        If ``shots`` is provided, the AerSimulator is used to sample
        measurement outcomes; otherwise a statevector simulation is used.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # Deterministic statevector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        else:
            sim = AerSimulator(method="statevector", shots=shots, seed_simulator=seed)
            for values in parameter_sets:
                bound = self._bind(values)
                job = sim.run(bound)
                result = job.result()
                counts = result.get_counts(bound)
                # Convert counts to expectation values
                exp_vals = []
                for obs in observables:
                    # Expectation value via Pauli expectation
                    exp = sum(
                        (1 if bit == "0" else -1) * count
                        for bit, count in counts.items()
                    ) / shots
                    exp_vals.append(complex(exp))
                results.append(exp_vals)
        return results

    @staticmethod
    def default_hybrid_circuit(n_wires: int = 4) -> QuantumCircuit:
        """Return a quantum circuit with classical encoding and a Quantum‑NAT style layer."""
        return HybridCircuit(n_wires)


__all__ = ["FastHybridEstimator", "HybridCircuit"]
