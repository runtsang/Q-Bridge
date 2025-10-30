import numpy as np
from collections.abc import Iterable, Sequence
from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QFCModel:
    """Quantum neural network with a 4‑qubit variational circuit.
    The circuit is fully parametrised; the first n_wires parameters encode the
    classical input via Ry gates, the remaining parameters drive the variational
    layer (RX, RZ, CNOT).  Measurements are taken in the Pauli‑Z basis."""
    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_wires)
        # Classical encoding: Ry gates with symbolic parameters
        enc_params = [Parameter(f"enc_rz_{i}") for i in range(self.n_wires)]
        for i, p in enumerate(enc_params):
            qc.ry(p, i)
        # Variational layer: RX and RZ gates with symbolic parameters
        var_params_rx = [Parameter(f"var_rx_{i}") for i in range(self.n_wires)]
        var_params_rz = [Parameter(f"var_rz_{i}") for i in range(self.n_wires)]
        for i, (p_rx, p_rz) in enumerate(zip(var_params_rx, var_params_rz)):
            qc.rx(p_rx, i)
            qc.rz(p_rz, i)
        # Entangling gates
        qc.cnot(0, 1)
        qc.cnot(1, 2)
        qc.cnot(2, 3)
        return qc

    def evaluate(self, x: Sequence[float]) -> np.ndarray:
        """Evaluate the circuit for a single input vector x.
        Returns a 1‑D array of Pauli‑Z expectation values for each qubit."""
        if len(x)!= self.n_wires:
            raise ValueError("Input length must match number of qubits.")
        bind_dict = {f"enc_rz_{i}": angle for i, angle in enumerate(x)}
        qc = self._circuit.copy()
        qc.assign_parameters(bind_dict, inplace=True)
        # Set variational parameters to zero for a pure forward pass
        for i in range(self.n_wires):
            qc.rx(0.0, i)
            qc.rz(0.0, i)
        state = Statevector.from_instruction(qc)
        return np.array([state.expectation_value('Z', qubits=[i]) for i in range(self.n_wires)])

    @property
    def circuit(self) -> QuantumCircuit:
        """Return the base parametrised circuit for use with FastEstimator."""
        return self._circuit

class FastEstimator:
    """Estimator that evaluates expectation values of observables for a parametrised
    quantum circuit.  Supports optional Gaussian shot noise."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(
                    rng.normal(np.real(v), max(1e-6, 1 / shots)),
                    rng.normal(np.imag(v), max(1e-6, 1 / shots))
                )
                for v in row
            ]
            noisy.append(noisy_row)
        return noisy
