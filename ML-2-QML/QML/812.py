"""Advanced hybrid estimator for quantum circuits with noise simulation and hybrid loss."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error

class FastBaseEstimator:
    """Estimator for parametrized quantum circuits with optional noise simulation and hybrid loss."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        noise: bool = False,
        shots: int | None = None,
        seed: int | None = None,
        backend: str = "aer_simulator",
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.noise = noise
        self.shots = shots
        self.seed = seed
        self.backend = backend
        self.simulator = AerSimulator(seed_simulator=seed, shots=shots if shots else 1024)
        if noise:
            self._add_noise()

    def _add_noise(self) -> None:
        """Add a simple depolarizing noise model to the simulator."""
        noise_model = NoiseModel()
        error = depolarizing_error(0.01, 1)
        for gate in self.circuit.gates:
            noise_model.add_all_qubit_quantum_error(error, gate[0])
        self.simulator.set_options(noise_model=noise_model)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind(params)
            job = self.simulator.run(circ)
            result = job.result()
            if self.shots:
                # Use measurement results to estimate expectation
                counts = result.get_counts(circ)
                exp_val = 0.0
                for bitstring, count in counts.items():
                    exp_val += self._bitstring_expectation(bitstring, observables) * count
                exp_val /= sum(counts.values())
                results.append([exp_val])
            else:
                state = Statevector.from_instruction(circ)
                exp_val = [state.expectation_value(obs) for obs in observables]
                results.append(exp_val)
        return results

    def _bitstring_expectation(self, bitstring: str, observables: Iterable[Operator]) -> complex:
        """Compute expectation value of observables given a classical bitstring."""
        state = Statevector.from_label(bitstring)
        return sum(state.expectation_value(obs) for obs in observables)

    def hybrid_loss(
        self,
        classical_outputs: List[float],
        quantum_outputs: List[complex],
        weight: float = 0.5,
    ) -> float:
        """Return weighted sum of classical and quantum outputs."""
        return weight * sum(classical_outputs) / len(classical_outputs) + \
               (1 - weight) * sum(quantum_outputs) / len(quantum_outputs)

__all__ = ["FastBaseEstimator"]
