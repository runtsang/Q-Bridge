"""Hybrid fully‑connected layer implemented with a parameterized quantum circuit.

The class mirrors the structure of the original FCL example but extends it
with batch‑wise evaluation, arbitrary observables and optional Gaussian
shot noise, following the FastBaseEstimator design.  The API is
compatible with the classical counterpart: ``HybridFCL().evaluate(...)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridFCL:
    """
    Parameterized quantum circuit that can be used as a fully‑connected
    layer in a hybrid quantum‑classical workflow.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : Optional[QuantumCircuit], default None
        Backend used for state‑vector simulation; if None, Aer qasm simulator
        is used for sampling when shots are requested.
    shots : int, default 1024
        Number of shots for sampling; if None, ideal state‑vector evaluation
        is performed.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: Optional[QuantumCircuit] = None,
        shots: int | None = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Create a parameterized circuit with two parameters per qubit:
        # an "input" and a "weight" parameter.
        self._circuit = QuantumCircuit(n_qubits)
        self.input_params: List[Parameter] = []
        self.weight_params: List[Parameter] = []
        for q in range(n_qubits):
            inp = Parameter(f"inp_{q}")
            wgt = Parameter(f"wgt_{q}")
            self.input_params.append(inp)
            self.weight_params.append(wgt)
            self._circuit.h(q)
            self._circuit.ry(inp, q)
            self._circuit.rx(wgt, q)
        self._circuit.measure_all()

        # Store the list of all parameters for binding convenience.
        self._parameter_list = self.input_params + self.weight_params

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """
        Bind a sequence of parameter values to the circuit.

        The values must be ordered as [input_0, weight_0, input_1, weight_1,...].
        """
        if len(parameter_values)!= len(self._parameter_list):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameter_list, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable over a batch of
        parameter sets.  If ``shots`` is provided, Gaussian noise with
        variance 1/shots is added to each expectation value to mimic
        shot‑noise effects.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Quantum operators whose expectation values are to be computed.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter sequences matching the circuit parameters.
        shots : Optional[int], default None
            Number of shots for sampling; if None, ideal state‑vector
            evaluation is performed.
        seed : Optional[int], default None
            Random seed for reproducible noise.

        Returns
        -------
        List[List[complex]]
            Outer list indexed by parameter set, inner list by observable.
        """
        observables = list(observables) or [SparsePauliOp.from_list([("I" * self.n_qubits, 1)])]
        results: List[List[complex]] = []

        # Ideal state‑vector evaluation for deterministic baseline
        for params in parameter_sets:
            bound_circuit = self._bind(params)
            state = Statevector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        # Sample‑based evaluation with shot‑noise
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for params in parameter_sets:
            bound_circuit = self._bind(params)
            job = execute(
                bound_circuit,
                backend=self.backend,
                shots=shots,
                parameter_binds=[{p: float(v) for p, v in zip(self._parameter_list, params)}],
            )
            counts = job.result().get_counts(bound_circuit)
            probs = {state: cnt / shots for state, cnt in counts.items()}

            # Compute expectation for each observable via sampling
            row: List[complex] = []
            for obs in observables:
                exp_val = 0.0 + 0.0j
                for state_str, prob in probs.items():
                    # Convert binary string to computational basis state
                    state = int(state_str, 2)
                    # Eigenvalue of Pauli string for this basis state
                    eigen = 1.0
                    for idx, pauli in enumerate(obs.primitive.paulis):
                        if pauli == "I":
                            continue
                        bit = (state >> (self.n_qubits - idx - 1)) & 1
                        eigen *= 1 if (pauli == "X" and bit == 0) or (pauli == "Z" and bit == 0) else -1
                    exp_val += eigen * prob
                row.append(exp_val)
            # Add Gaussian shot noise
            noisy_row = [
                complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                        rng.normal(val.imag, max(1e-6, 1 / shots)))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridFCL"]
