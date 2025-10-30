from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class QuantumSelfAttention:
    """Quantum self‑attention circuit builder."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        return circ


class HybridBaseEstimator:
    """Hybrid estimator for quantum circuits with optional self‑attention and shot sampling."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        use_self_attention: bool = False,
        sa_params: dict | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.use_self_attention = use_self_attention
        self.shots = shots
        self.seed = seed
        if use_self_attention:
            if sa_params is None:
                raise ValueError("sa_params must be provided when use_self_attention is True")
            self.sa = QuantumSelfAttention(n_qubits=sa_params["n_qubits"])
            self.sa_rotation = sa_params["rotation_params"]
            self.sa_entangle = sa_params["entangle_params"]
        else:
            self.sa = None
        self.backend = Aer.get_backend("qasm_simulator")

    def _compose(self, params: Sequence[float]) -> QuantumCircuit:
        circ = self.circuit.copy()
        if self.use_self_attention:
            sa_circ = self.sa._build(self.sa_rotation, self.sa_entangle)
            circ = sa_circ.compose(circ, front=True)
        return circ.bind_parameters(dict(zip(circ.parameters, params)))

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circ = self._compose(params)
            if self.shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound_circ, self.backend, shots=self.shots, seed_simulator=self.seed)
                counts = job.result().get_counts(bound_circ)
                row = []
                for obs in observables:
                    # Simplified conversion: only Pauli‑Z expectation values are supported
                    if hasattr(obs, "data") and obs.data[0][0] == 1:
                        exp = sum(
                            (1 if bitstring[-1] == "0" else -1) * count
                            for bitstring, count in counts.items()
                        )
                        exp /= sum(counts.values())
                        row.append(exp)
                    else:
                        row.append(0 + 0j)
            results.append(row)
        return results


__all__ = ["HybridBaseEstimator"]
