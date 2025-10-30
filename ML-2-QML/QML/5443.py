"""Hybrid quantum estimator with a RealAmplitudes ansatz and domain‑wall swap‑test."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Callable

# --------------------------------------------------------------------------- #
#  Fast quantum evaluator (adapted from FastBaseEstimator.py – quantum version)
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Deterministic evaluator for a parametrized quantum circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
#  Domain‑wall swap‑test helper (adapted from Autoencoder.py quantum section)
# --------------------------------------------------------------------------- #

def domain_wall_swap(num_qubits: int, start: int, end: int) -> QuantumCircuit:
    """Apply X gates to qubits in the given range to create a domain wall."""
    qc = QuantumCircuit(num_qubits)
    for i in range(start, end):
        qc.x(i)
    return qc


# --------------------------------------------------------------------------- #
#  Hybrid quantum circuit
# --------------------------------------------------------------------------- #

class HybridQuantumCircuit:
    """Parameterized circuit that implements a RealAmplitudes ansatz,
    followed by a domain‑wall swap‑test, and a final measurement."""
    def __init__(self, latent_dim: int, trash_dim: int = 2) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.num_qubits = latent_dim + 2 * trash_dim + 1  # +1 for auxiliary
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz on latent+trash qubits
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)
        qc.compose(ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)
        qc.barrier()

        # Auxiliary qubit for swap‑test
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)

        # Domain‑wall on trash qubits
        dw_qc = domain_wall_swap(self.num_qubits, self.latent_dim + self.trash_dim, self.num_qubits)
        qc.compose(dw_qc, inplace=True)

        qc.measure(aux, cr[0])
        return qc

    def run(self, theta_params: Sequence[float]) -> np.ndarray:
        """Run the circuit on a statevector sampler and return the expectation."""
        sampler = StatevectorSampler()
        job = sampler.run(self._circuit, shots=1024, parameter_binds=[{p: v} for p, v in zip(self._circuit.parameters, theta_params)])
        result = job.result()
        counts = result.get_counts(self._circuit)
        probs = np.array(list(counts.values())) / 1024
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Convenience wrapper around FastBaseEstimator."""
        evaluator = FastBaseEstimator(self._circuit)
        return evaluator.evaluate(observables, parameter_sets)


__all__ = ["HybridQuantumCircuit", "FastBaseEstimator", "domain_wall_swap"]
