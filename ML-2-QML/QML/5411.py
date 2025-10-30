"""Hybrid estimator that evaluates quantum circuits and supports shot‑noise simulation."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from typing import Iterable, List, Sequence, Tuple

class QuanvCircuit:
    """Quantum filter emulating a 2×2 convolution via parameterised rotations."""
    def __init__(self, kernel_size: int = 2, backend: QuantumCircuit | None = None, shots: int = 100, threshold: float = 0.0) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(self.theta):
            self._circuit.rx(t, i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit for a single 2×2 patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: (np.pi if val > self.threshold else 0) for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

class QuanvolutionFilter:
    """Quantum quanvolution filter that applies a simple two‑qubit kernel to image patches."""
    def __init__(self, kernel_size: int = 2, depth: int = 2, backend: QuantumCircuit | None = None, shots: int = 100) -> None:
        self.n_wires = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.depth = depth

    def run(self, data: np.ndarray) -> np.ndarray:
        """Apply the quantum kernel to each 2×2 patch and return measurement results."""
        data = data.reshape(-1, self.n_wires)
        results = []
        for patch in data:
            qc = QuantumCircuit(self.n_wires)
            for i, val in enumerate(patch):
                qc.ry(val, i)
            for _ in range(self.depth):
                for i in range(self.n_wires - 1):
                    qc.cz(i, i + 1)
            qc.measure_all()
            job = execute(qc, self.backend, shots=self.shots)
            res = job.result().get_counts(qc)
            # Convert counts to expectation of Z on each qubit
            expectation = []
            for i in range(self.n_wires):
                prob_1 = sum(cnt for bitstring, cnt in res.items() if bitstring[self.n_wires - 1 - i] == '1') / self.shots
                expectation.append(1 - 2 * prob_1)
            results.append(expectation)
        return np.array(results)

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Return a layered ansatz with encoding, variational parameters, and observables."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, q in zip(encoding, range(num_qubits)):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

class FastBaseEstimatorGen302:
    """Quantum estimator that evaluates parametrised circuits and supports shot‑noise simulation."""
    def __init__(self, circuit: QuantumCircuit, backend: QuantumCircuit | None = None) -> None:
        self._circuit = circuit
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {param: val for param, val in zip(self._circuit.parameters, parameter_values)}
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound_qc = self._bind(params)
            job = execute(bound_qc, self.backend, shots=shots or 1024)
            result = job.result()
            state = Statevector.from_instruction(bound_qc)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(float(val.real), max(1e-6, 1 / shots))
                + 1j * rng.normal(float(val.imag), max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    def predict(self, parameter_values: Sequence[float]) -> Statevector:
        """Return the statevector after applying the circuit with given parameters."""
        bound_qc = self._bind(parameter_values)
        return Statevector.from_instruction(bound_qc)

__all__ = ["FastBaseEstimatorGen302", "QuanvCircuit", "QuanvolutionFilter", "build_classifier_circuit"]
