import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter
from typing import Iterable, Sequence, List

class HybridFCL:
    """
    Quantum implementation of a fully‑connected layer.  The circuit
    mirrors the classical version but uses a single qubit with an
    RX rotation whose angle is supplied as the parameter set.  The
    class exposes the same ``run`` and ``evaluate`` interface as its
    classical counterpart.
    """
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots
        self.n_qubits = n_qubits

        # Build a reusable circuit template
        self._template = QuantumCircuit(n_qubits)
        self._template.h(range(n_qubits))
        self._template.barrier()
        self._template.ry(Parameter("theta"), range(n_qubits))
        self._template.measure_all()

    def _run_once(self, theta: float) -> float:
        """Execute the circuit for a single theta and return expectation."""
        circ = self._template.assign_parameters({"theta": theta})
        job = execute(circ, self.backend, shots=self.shots)
        result = job.result().get_counts(circ)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return float(expectation)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a single set of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Sequence of parameters (should match the number of qubits).

        Returns
        -------
        np.ndarray
            Mean expectation over the supplied parameters.
        """
        expectations = [self._run_once(float(theta)) for theta in thetas]
        return np.mean(expectations)

    def evaluate(
        self,
        thetas_list: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Batch evaluate many parameter sets with optional shot noise.

        Parameters
        ----------
        thetas_list : Sequence[Sequence[float]]
            List of parameter sequences to evaluate.
        shots : int, optional
            If provided, Gaussian shot noise with variance 1/shots is added.
        seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the expectation
            value for the corresponding parameter set.
        """
        results: List[List[float]] = []
        if shots is None:
            for params in thetas_list:
                results.append([float(self.run(params))])
            return results

        rng = np.random.default_rng(seed)
        for params in thetas_list:
            mean = float(self.run(params))
            noisy = rng.normal(mean, max(1e-6, 1 / shots))
            results.append([float(noisy)])
        return results
