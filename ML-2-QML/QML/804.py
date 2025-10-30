"""Quantum estimator with support for shot simulation, noise, and hybrid training.

Features
--------
- Device selection (Aer, Pennylane default, or any Qiskit backend).
- Shot-based expectation value estimation.
- Noise model integration.
- Compatibility with Pennylane for parameter‑shift gradient computation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
import pennylane as qml
from pennylane import qiskit as qml_qiskit
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Backend

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized quantum circuit.
    backend : Backend | str, optional
        Backend to use for simulation. Defaults to Aer.get_backend('statevector_simulator').
    shots : int | None, optional
        Number of shots for expectation estimation. If None, use statevector evaluation.
    noise_model : Optional[Backend], optional
        Noise model to apply. Only used when shots is not None.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Union[Backend, str] | None = None,
        shots: int | None = None,
        noise_model: Optional[Backend] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

        if backend is None:
            backend = Aer.get_backend("statevector_simulator") if shots is None else Aer.get_backend("qasm_simulator")
        self.backend = backend if isinstance(backend, Backend) else Aer.get_backend(backend)
        self.shots = shots
        self.noise_model = noise_model

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _execute(self, bound_circuit: QuantumCircuit) -> np.ndarray:
        if self.shots is None:
            state = Statevector.from_instruction(bound_circuit)
            return state
        else:
            job = execute(
                bound_circuit,
                backend=self.backend,
                shots=self.shots,
                noise_model=self.noise_model,
            )
            return job.result().get_counts(bound_circuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                state = self._execute(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                counts = self._execute(bound)
                total = sum(counts.values())
                probs = {k: v / total for k, v in counts.items()}
                # For simplicity, we assume observables are Pauli strings represented as qiskit.quantum_info.Operator
                row = []
                for obs in observables:
                    exp_val = 0.0
                    for bitstring, prob in probs.items():
                        # compute eigenvalue of obs on basis state
                        eigen = obs.eigenvalues(bitstring)  # placeholder
                        exp_val += prob * eigen
                    row.append(exp_val)
            results.append(row)

        return results

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute gradients using the parameter‑shift rule via PennyLane."""
        # Convert the Qiskit circuit to a PennyLane circuit
        pl_circuit = qml_qiskit.QiskitCircuit(self._circuit)
        dev = qml.device("default.qubit", wires=pl_circuit.num_wires)

        def qnode(*params):
            pl_circuit.apply(params)
            return [qml.expval(obs) for obs in observables]

        grad_fn = qml.grad(lambda *params: qnode(*params))
        grads = []
        for values in parameter_sets:
            grads.append(grad_fn(*values))
        return grads

__all__ = ["FastBaseEstimator"]
