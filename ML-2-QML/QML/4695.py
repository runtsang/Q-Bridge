"""Hybrid estimator for quantum circuits (qiskit) with shot‑noise simulation
and a toy quantum fully‑connected layer."""
from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Optional
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------- utility layer -----------------------------
class QuantumFCL:
    """Simple parameterised quantum circuit for a toy fully‑connected layer."""
    def __init__(self, n_qubits: int, shots: int = 100) -> None:
        self._circuit = QuantumCircuit(n_qubits)
        theta = self._circuit.parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{"theta": theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()], dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])


# --------------------------- main estimator ----------------------------
class HybridEstimator:
    """Evaluate a qiskit circuit for sequences of parameters."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        If ``shots`` is provided, Gaussian noise simulating shot‑noise is
        applied to the deterministic expectation values.
        """
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(float(exp), max(1e-6, 1 / shots))) for exp in row]
                noisy.append(noisy_row)
            return noisy
        return results

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the toy quantum fully‑connected layer."""
        fcl = QuantumFCL(n_qubits=1, shots=100)
        return fcl.run(thetas)

    def evaluate_sequence(self, sentence) -> None:
        """
        Placeholder for sequence evaluation if the circuit is a wrapped
        quantum LSTM.  Implementation depends on the specific LSTM
        quantum package used (e.g., torchquantum or custom circuit).
        """
        raise NotImplementedError("Sequence evaluation not implemented for this circuit.")


__all__ = ["HybridEstimator", "QuantumFCL"]
