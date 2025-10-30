"""FastBaseEstimator for Qiskit quantum circuits with optional shot‑sampling.

The class evaluates a parameterised ``QuantumCircuit`` for a list of parameter sets,
computing expectation values of Pauli observables.  Finite‑shot sampling is
performed when ``shots`` is supplied, otherwise a deterministic state‑vector
backend is used.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Pauli, Statevector, BaseOperator

class FastBaseEstimator:
    """
    Evaluate expectation values of a parameterised QuantumCircuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        A circuit that contains ``Parameter`` objects.  The number of parameters
        must match the length of each parameter set supplied to :meth:`evaluate`.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._original = circuit
        self._parameters = list(circuit.parameters)
        if not self._parameters:
            raise ValueError("Circuit must contain at least one Parameter.")
        # Pre‑compile the circuit for speed
        self._compiled = transpile(circuit, optimization_level=3)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._original.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : iterable of Pauli or BaseOperator
            Each element is a Hermitian operator expressed as a Pauli string.
        parameter_sets : sequence of parameter vectors
            Each vector is bound to the circuit in order.
        shots : int, optional
            If supplied, a qasm simulator with the given shot count is used.
            The returned values are the mean of the sampled measurement outcomes,
            which introduces finite‑shot noise.
        seed : int, optional
            Random seed for the qasm simulator; ignored when ``shots`` is None.

        Returns
        -------
        List[List[complex]]
            A table where rows correspond to parameter sets and columns to observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        # Choose backend
        if shots is None:
            backend = AerSimulator(method="statevector")
        else:
            backend = AerSimulator(method="qasm", shots=shots, seed_simulator=seed)

        for values in parameter_sets:
            circ = self._bind(values)
            circ = transpile(circ, backend, optimization_level=3)
            if shots is None:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Build measurement circuit
                meas_circ = circ.copy()
                meas_circ.measure_all()
                qobj = assemble(meas_circ, backend=backend)
                result = backend.run(qobj).result()
                counts = result.get_counts()
                exp_vals: List[complex] = []
                for obs in observables:
                    if isinstance(obs, Pauli):
                        pauli_str = str(obs)
                        # Compute parity for each bitstring
                        exp = 0.0
                        for bitstr, freq in counts.items():
                            # bitstring is in little‑endian order
                            bits = [int(b) for b in bitstr[::-1]]
                            parity = 1
                            for idx, p in enumerate(pauli_str):
                                if p!= 'I':
                                    if bits[idx] == 1:
                                        parity *= -1
                            exp += parity * freq
                        exp_vals.append(exp / sum(counts.values()))
                    else:
                        # For non‑Pauli operators, fall back to statevector evaluation
                        state = Statevector.from_instruction(circ)
                        exp_vals.append(state.expectation_value(obs))
                row = exp_vals
            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
