import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import Parameter
from collections.abc import Iterable, Sequence
from typing import List

class SelfAttentionGen330:
    """Quantum self‑attention with expectation‑value evaluation."""
    def __init__(self, n_qubits: int = 4, backend=None) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self._template = self._create_template()

    def _create_template(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        # rotation parameters
        self.rotation_params = [Parameter(f'rot_{i}') for i in range(3 * self.n_qubits)]
        # entanglement parameters
        self.entangle_params = [Parameter(f'ent_{i}') for i in range(self.n_qubits - 1)]
        for i in range(self.n_qubits):
            circ.rx(self.rotation_params[3 * i], i)
            circ.ry(self.rotation_params[3 * i + 1], i)
            circ.rz(self.rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(self.entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def _bind_parameters(self, param_values: Sequence[float]) -> QuantumCircuit:
        total_params = len(self.rotation_params) + len(self.entangle_params)
        if len(param_values)!= total_params:
            raise ValueError("Parameter count mismatch.")
        mapping = {}
        for p, v in zip(self.rotation_params + self.entangle_params, param_values):
            mapping[p] = v
        return self._template.assign_parameters(mapping, inplace=False)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """Execute the circuit and return raw measurement counts."""
        circuit = self._create_measurement_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def _create_measurement_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind_parameters(values)
            state = Statevector.from_instruction(circ)
            results.append([state.expectation_value(obs) for obs in observables])
        return results

__all__ = ["SelfAttentionGen330"]
