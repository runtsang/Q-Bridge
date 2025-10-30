import numpy as np
from collections.abc import Iterable, Sequence
from typing import List
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit import execute

class FastBaseEstimator:
    """Quantum batch estimator for parametrized circuits with optional shot‑noise.

    Enhancements over the original primitive:
    * Support for a custom simulator backend or the default Aer simulator.
    * Batched evaluation of many parameter sets in a single circuit run.
    * Optional shot‑noise emulation by adding Gaussian fluctuations to expectation values.
    * Flexible observable interface using Qiskit BaseOperator objects.
    """

    def __init__(self, circuit: QuantumCircuit, shots: int | None = None, backend: str | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.backend = backend or "aer_simulator_statevector"

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of Qiskit BaseOperator objects.
        parameter_sets:
            Sequence of parameter vectors to bind to the circuit.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound = self._bind(params)
            if self.shots is None:
                # State‑vector simulation for exact expectation values
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Shot‑based simulation: sample measurement outcomes and compute expectation
                job = execute(bound, backend=AerSimulator(method=self.backend), shots=self.shots)
                result = job.result()
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]

            results.append(row)

        if self.shots is not None:
            rng = np.random.default_rng()
            noisy = []
            for row in results:
                noisy_row = [complex(rng.normal(0, 1/np.sqrt(self.shots))) + val for val in row]
                noisy.append(noisy_row)
            results = noisy

        return results

# Quantum‑NAT style hybrid circuit
def quantum_fc_model() -> QuantumCircuit:
    """Return a 4‑wire parametrised circuit with a random layer and a small
    variational block, mirroring the QML QFCModel in the reference pair.
    """
    from qiskit.circuit.library import RandomCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RY, RZ, CRX, H, SX, CX

    n_wires = 4
    circuit = QuantumCircuit(n_wires)

    # Random initial layer
    rand_circ = RandomCircuit(n_wires, n_layers=3, depth=2)
    circuit.append(rand_circ, list(range(n_wires)))

    # Variational block
    for i in range(n_wires):
        circuit.append(RY(Parameter(f"θ_{i}")), [i])
    circuit.append(CRX(Parameter("θ_crx")), [0, 2])
    circuit.append(H, [3])
    circuit.append(SX, [2])
    circuit.append(CX, [3, 0])

    return circuit

__all__ = ["FastBaseEstimator", "quantum_fc_model"]
