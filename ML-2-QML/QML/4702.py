"""Hybrid quantum classifier combining incremental encoding, variational layers,
quantum self‑attention, and fast expectation evaluation."""
from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Sequence, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

# ----- Quantum self‑attention helper ------------------------------------------
class QuantumSelfAttention:
    """Quantum block mimicking classical self‑attention."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def build(self, circuit: QuantumCircuit, params: Sequence[float], offset: int) -> int:
        """Inject self‑attention gates into `circuit` starting at `offset` in `params`."""
        n = self.n_qubits
        # Rotation gates
        for i in range(n):
            circuit.rx(params[offset + 3 * i], i)
            circuit.ry(params[offset + 3 * i + 1], i)
            circuit.rz(params[offset + 3 * i + 2], i)
        offset += 3 * n
        # Entangling CRX gates
        for i in range(n - 1):
            circuit.crx(params[offset + i], i, i + 1)
        offset += n - 1
        return offset

# ----- Fast estimator for quantum circuits ------------------------------------
class FastBaseEstimator:
    """Compute expectation values of Pauli observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        if len(param_vals)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_vals))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        obs = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(ob) for ob in obs]
            results.append(row)
        return results

# ----- Hybrid quantum classifier -----------------------------------------------
class HybridClassifier:
    """Quantum classifier with optional quantum self‑attention block."""
    def __init__(self, num_qubits: int, depth: int, use_attention: bool = True):
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_attention = use_attention
        self.circuit, self.encoding, self.weight_sizes, self.observables = self._build_circuit()
        self.estimator = FastBaseEstimator(self.circuit)

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[int], Iterable[int], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = QuantumCircuit(self.num_qubits)
        # Encoding
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)
        # Variational layers with optional self‑attention
        w_idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[w_idx], qubit)
                w_idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
            if self.use_attention:
                # Self‑attention parameters
                phi = ParameterVector("phi", 3 * self.num_qubits)
                psi = ParameterVector("psi", self.num_qubits - 1)
                param_seq = list(phi) + list(psi)
                # Inject gates
                offset = 0
                for i in range(self.depth):
                    offset = QuantumSelfAttention(self.num_qubits).build(circuit, param_seq, offset)
        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        weight_sizes = [len(circuit.parameters)]
        encoding_indices = list(range(self.num_qubits))
        return circuit, encoding_indices, weight_sizes, observables

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Return expectation values for each parameter set."""
        return self.estimator.evaluate(self.observables, parameter_sets)

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    use_attention: bool = True,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Return a hybrid quantum circuit and metadata compatible with the original API."""
    model = HybridClassifier(num_qubits, depth, use_attention)
    return model.circuit, model.encoding, model.weight_sizes, model.observables

__all__ = ["HybridClassifier", "build_classifier_circuit", "FastBaseEstimator"]
