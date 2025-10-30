import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

class QuantumSelfAttention:
    """Quantum self‑attention block mirroring the classical counterpart."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def bind(self, params: Sequence[float], rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = self._build(rotation_params, entangle_params)
        mapping = dict(zip(circ.parameters, params))
        return circ.assign_parameters(mapping, inplace=False)

class FastHybridEstimator:
    """Unified estimator executing a parametrised quantum circuit with optional self‑attention."""
    def __init__(self,
                 base_circuit: QuantumCircuit,
                 *,
                 attention_circuit: QuantumCircuit | None = None,
                 backend=None,
                 shots: int = 1024) -> None:
        self.base_circuit = base_circuit
        self.attention_circuit = attention_circuit
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.base_params = len(self.base_circuit.parameters)
        self.attn_params = len(self.attention_circuit.parameters) if self.attention_circuit else 0

    def _bind(self, circuit: QuantumCircuit, params: Sequence[float]) -> QuantumCircuit:
        mapping = dict(zip(circuit.parameters, params))
        return circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []

        for param_set in parameter_sets:
            idx = 0
            if self.attn_params:
                attn_vals = param_set[idx: idx + self.attn_params]
                idx += self.attn_params
            else:
                attn_vals = []

            base_vals = param_set[idx: idx + self.base_params]
            idx += self.base_params

            circ = self._bind(self.base_circuit, base_vals)

            if self.attention_circuit:
                attn_circ = self._bind(self.attention_circuit, attn_vals)
                attn_state = Statevector.from_instruction(attn_circ)
                probs = np.abs(attn_state.data)**2
                circ = self._bind(circ, probs[:self.base_params])

            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result().get_counts()

            row = []
            for obs in observables:
                state = Statevector.from_instruction(circ)
                exp = state.expectation_value(obs)
                row.append(exp)
            results.append(row)

        return results

__all__ = ["FastHybridEstimator"]
