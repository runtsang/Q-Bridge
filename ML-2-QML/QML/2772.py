"""EstimatorQNNGen025: quantum estimator that evaluates a 2‑qubit variational circuit.

This class extends the original EstimatorQNN by adding two entangling
layers and a richer set of observables.  It follows the FastBaseEstimator
API, providing a batched evaluate method that returns expectation
values for each supplied parameter set.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers.aer import AerSimulator
from typing import Iterable, Sequence, List, Any, Tuple

class EstimatorQNNGen025:
    """
    Quantum estimator that evaluates a 2‑qubit variational circuit.
    The circuit is parameterized by four real parameters:
    (theta1, phi1, theta2, phi2).  The first two are used as input
    and weight parameters respectively.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: Iterable[SparsePauliOp],
        input_params: Sequence[Parameter],
        weight_params: Sequence[Parameter],
        estimator: Any = None,
    ) -> None:
        self.circuit = circuit
        self.observables = list(observables)
        self.input_params = list(input_params)
        self.weight_params = list(weight_params)
        self.estimator = estimator or AerSimulator(method="statevector")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.input_params) + len(self.weight_params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.input_params + self.weight_params, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circ = self._bind(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    @classmethod
    def create_default(cls) -> "EstimatorQNNGen025":
        """
        Helper to construct a default 2‑qubit variational circuit with
        two layers and a set of observables.
        """
        # Define parameters
        theta1 = Parameter("theta1")  # input
        phi1   = Parameter("phi1")    # weight
        theta2 = Parameter("theta2")
        phi2   = Parameter("phi2")

        qc = QuantumCircuit(2)
        # Layer 1
        qc.ry(theta1, 0)
        qc.rz(phi1, 0)
        qc.ry(theta2, 1)
        qc.rz(phi2, 1)
        qc.cx(0, 1)
        # Layer 2
        qc.ry(theta1, 0)
        qc.rz(phi1, 0)
        qc.ry(theta2, 1)
        qc.rz(phi2, 1)
        qc.cx(1, 0)

        # Observables
        obs_y0 = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
        obs_z1 = SparsePauliOp.from_list([("I" * (qc.num_qubits - 1) + "Z", 1)])
        return cls(
            circuit=qc,
            observables=[obs_y0, obs_z1],
            input_params=[theta1, theta2],
            weight_params=[phi1, phi2],
            estimator=AerSimulator(method="statevector"),
        )

__all__ = ["EstimatorQNNGen025"]
