"""Quantum estimator that evaluates a parameterized circuit with a quanvolution filter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli, Operator
from qiskit_aer import AerSimulator

# --------------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter:
    """
    A quantum implementation of the 2×2 patch filter.  Each patch is encoded
    into a 4‑qubit circuit using Ry rotations, then an entangling layer
    is applied.  The filter returns a statevector that can be used for
    expectation‑value evaluation.
    """
    def __init__(self, n_entangling_layers: int = 1, seed: Optional[int] = None) -> None:
        self.n_entangling = n_entangling_layers
        self.seed = seed

    def _encode(self, qc: QuantumCircuit, patch: Sequence[float], wires: Sequence[int]) -> None:
        """Apply Ry gates that encode the patch values."""
        for idx, val in enumerate(patch):
            qc.ry(val, wires[idx])

    def _entangle(self, qc: QuantumCircuit, wires: Sequence[int]) -> None:
        """Apply a simple entangling pattern."""
        for _ in range(self.n_entangling):
            for w in wires:
                qc.h(w)
                qc.cx(w, wires[(w + 1) % len(wires)])

    def circuit(self, patch: Sequence[float]) -> QuantumCircuit:
        """Return a parameter‑free circuit for a single patch."""
        wires = list(range(4))
        qc = QuantumCircuit(4)
        self._encode(qc, patch, wires)
        self._entangle(qc, wires)
        return qc

# --------------------------------------------------------------------------- #
# Hybrid quantum estimator
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """
    Evaluate a collection of parameter sets on a quantum circuit that
    implements a quanvolution filter.  Observables are given as
    ``Operator`` instances (e.g. Pauli matrices).  The estimator can run
    with finite shots to mimic measurement noise.
    """
    def __init__(self, filter: QuantumQuanvolutionFilter) -> None:
        self.filter = filter
        self.simulator = AerSimulator()

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Parameters
        ----------
        observables : iterable of qiskit.quantum_info.Operator
            Operators for which the expectation value is computed.
        parameter_sets : sequence of parameter lists
            Each list contains 28×28 values that are unpacked into 2×2 patches.
        shots : int | None
            Number of measurement shots.  If ``None`` the statevector is used
            directly.  This keeps the interface identical to the classical
            estimator.
        seed : int | None
            RNG seed for the simulator.

        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets, columns to observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            # Split the flattened image into 2×2 patches
            patches = [params[i:i + 4] for i in range(0, len(params), 4)]
            row: List[complex] = []

            for obs in observables:
                exp_val: complex = 0.0 + 0.0j

                for patch in patches:
                    qc = self.filter.circuit(patch)
                    # If shots are requested we perform a measurement simulation
                    if shots is not None:
                        qc.measure_all()
                        transpiled = transpile(qc, self.simulator)
                        result = self.simulator.run(
                            transpiled,
                            shots=shots,
                            seed_simulator=seed,
                            seed_transpiler=seed,
                        ).result()
                        counts = result.get_counts()
                        state = Statevector.from_counts(counts)
                    else:
                        state = Statevector.from_instruction(qc)

                    exp_val += state.expectation_value(obs)

                row.append(exp_val)
            results.append(row)

        return results

__all__ = ["HybridEstimator", "QuantumQuanvolutionFilter"]
