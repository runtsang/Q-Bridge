import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit

class HybridFCL:
    """
    Quantum implementation of the hybrid layer.

    Parameters
    ----------
    mode : str
        'fc' for a fully‑connected style circuit, 'conv' for a quanvolution
        filter.  The implementation is a drop‑in replacement for the
        quantum examples in the seeds.
    n_qubits : int
        Number of qubits used in the circuit.
    backend : qiskit.providers.Backend
        Backend to execute the circuit.
    shots : int
        Number of shots for sampling.
    threshold : float, optional
        Threshold for the convolutional mode.
    """
    def __init__(self,
                 mode: str = "fc",
                 n_qubits: int = 1,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 0.5) -> None:
        self.mode = mode
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        if mode == "fc":
            self._circuit = QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()
        elif mode == "conv":
            self._circuit = QuantumCircuit(n_qubits)
            self.thetas = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
            for i in range(n_qubits):
                self._circuit.rx(self.thetas[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(n_qubits, 2)
            self._circuit.measure_all()
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter values for the circuit.  For the 'fc' mode a single
            value is used; for 'conv' a value per qubit is required.

        Returns
        -------
        np.ndarray
            Array containing the expectation value of Pauli‑Z on all
            measured qubits, averaged over shots.
        """
        if self.mode == "fc":
            param_bind = {self.theta: thetas[0]}
        else:
            if len(thetas)!= self.n_qubits:
                raise ValueError("Parameter count mismatch for bound circuit.")
            param_bind = dict(zip(self.thetas, thetas))

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

class FastBaseEstimator:
    """
    Quantitative estimator that evaluates expectation values of
    BaseOperator observables for a batch of parameter sets.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["HybridFCL", "FastBaseEstimator"]
