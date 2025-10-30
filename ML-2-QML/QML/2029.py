"""Quantum implementation of a fully connected layer with gradient support."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from typing import Iterable, Sequence

class FCL:
    """
    Parameterized quantum circuit that emulates a fully connected layer.
    Supports batch execution, noise simulation, and parameter‑shift gradient.
    """
    def __init__(self, n_qubits: int = 1, backend: Backend | None = None,
                 shots: int = 1000, noise_model=None, transpile_options=None):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        backend : Backend, optional
            Quantum backend to use; defaults to Aer qasm simulator.
        shots : int
            Number of shots for each execution.
        noise_model : qiskit.providers.models.NoiseModel, optional
            Noise model to apply to the simulator.
        transpile_options : dict, optional
            Transpilation options passed to ``transpile``.
        """
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.noise_model = noise_model
        self.transpile_options = transpile_options or {}
        self.theta = Parameter("θ")
        self._build_circuit()
        self._compiled_circuit = None

    def _build_circuit(self):
        """Construct a simple parametrized circuit."""
        self.circuit = QuantumCircuit(self.n_qubits)
        # Layer 1: H gates
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        # Layer 2: Parameterized Ry rotations
        for q in range(self.n_qubits):
            self.circuit.ry(self.theta, q)
        self.circuit.barrier()
        # Layer 3: Entangling CNOT chain
        for q in range(self.n_qubits - 1):
            self.circuit.cx(q, q + 1)
        self.circuit.barrier()
        # Measurement
        self.circuit.measure_all()

    def compile(self, thetas: Iterable[float]):
        """Compile the circuit with the given theta value(s)."""
        if isinstance(thetas, (list, tuple, np.ndarray)):
            bind = {self.theta: thetas[0]}
        else:
            bind = {self.theta: thetas}
        bound = self.circuit.bind_parameters(bind)
        self._compiled_circuit = transpile(bound, self.backend, **self.transpile_options)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for a single theta value or a batch.
        Returns expectation value of the computational basis measurement.
        """
        if isinstance(thetas, (list, tuple, np.ndarray)) and len(thetas) > 1:
            # Batch execution: run each theta sequentially
            results = []
            for th in thetas:
                self.compile(th)
                job = execute(self._compiled_circuit, self.backend,
                              shots=self.shots, noise_model=self.noise_model)
                result = job.result()
                exp = self._expectation(result)
                results.append(exp)
            return np.array(results)
        else:
            self.compile(thetas)
            job = execute(self._compiled_circuit, self.backend,
                          shots=self.shots, noise_model=self.noise_model)
            result = job.result()
            return np.array([self._expectation(result)])

    def _expectation(self, result):
        """Compute expectation value from measurement counts."""
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        # Interpret bitstring as integer value
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        return np.sum(states * probs)

    def parameter_shift_gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Estimate the gradient of the expectation value with respect to theta
        using the parameter‑shift rule.
        """
        shift = np.pi / 2
        grad_plus = self.run(thetas + shift)
        grad_minus = self.run(thetas - shift)
        return 0.5 * (grad_plus - grad_minus)

__all__ = ["FCL"]
