import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
import numpy as np
from typing import Callable, List, Tuple

class FastHybridEstimator:
    """Hybrid estimator that evaluates a parametrized quantum circuit and optional
    classical post‑processing.  It mirrors the classical implementation but
    operates on Qiskit circuits and supports shot‑noise, random parameter
    initialization, and a simple quanvolution filter implemented as a
    small sub‑circuit.

    The API is deliberately symmetric to the classical FastHybridEstimator:
    - ``evaluate`` accepts observables and a list of parameter vectors.
    - ``shots`` adds Gaussian shot‑noise to the returned expectation values.
    """

    def __init__(self, circuit: QuantumCircuit, backend: qiskit.providers.Backend | None = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("aer_simulator_statevector")

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, params))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each supplied parameter set.

        Parameters
        ----------
        observables:
            Iterable of Pauli operators or other BaseOperator instances.
            If None a single observable (Z on the first qubit) is used.
        parameter_sets:
            Sequence of parameter lists.  Each inner list is bound to the circuit
            before evaluation.
        shots:
            If provided, the returned values are perturbed with Gaussian noise
            with std = 1/√shots to mimic finite‑shot statistics.
        seed:
            Random seed for reproducibility of the noise.

        Returns
        -------
        list[list[complex]]:
            Nested list where each inner list contains the expectation value
            for the corresponding observable.
        """
        if observables is None:
            observables = [SparsePauliOp("Z")]
        if parameter_sets is None:
            parameter_sets = []

        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)

        for params in parameter_sets:
            bound = self._bind(params)
            state = Statevector.from_instruction(bound)
            row: List[complex] = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                # Add Gaussian noise per observable
                row = [rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots) for val in row]
            results.append(row)
        return results

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int = 1) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Construct a layered variational ansatz with data encoding and return metadata.

        Parameters
        ----------
        num_qubits:
            Number of qubits (also the dimensionality of the input).
        depth:
            Number of variational layers.

        Returns
        -------
        circuit:
            The constructed QuantumCircuit.
        encoding:
            List of ParameterVector objects used for data encoding.
        weights:
            List of ParameterVector objects representing variational parameters.
        observables:
            List of PauliZ operators on each qubit, kept for API parity.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables

    @staticmethod
    def build_quanvolution_filter(kernel_size: int = 2, shots: int = 100, threshold: float = 0.5) -> QuantumCircuit:
        """Return a small quantum circuit that acts as a quanvolution filter.

        The circuit encodes a 2×2 patch into a 4‑qubit register, applies a random
        Clifford layer, measures all qubits, and returns the average |1> probability.
        The filter is intentionally tiny so it can be tiled over an image grid.

        Parameters
        ----------
        kernel_size:
            Size of the square patch (must be 2 for the current implementation).
        shots:
            Number of shots for the simulation; used to add realistic statistical
            noise when the circuit is run on a real device.
        threshold:
            Classical threshold used when mapping pixel values to π rotations.
        """
        n_qubits = kernel_size ** 2
        circuit = QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]

        # Data encoding: rotate each qubit by its pixel value
        for i, t in enumerate(theta):
            circuit.rx(t, i)

        # Random Clifford layer (here we use a 2‑step random circuit)
        circuit += qiskit.circuit.random.random_circuit(n_qubits, 2, seed=42)

        circuit.measure_all()
        return circuit

__all__ = ["FastHybridEstimator"]
