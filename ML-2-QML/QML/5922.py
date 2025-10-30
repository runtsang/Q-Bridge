import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from typing import Iterable, Optional

class FCLHybrid:
    """
    Quantum implementation of the hybrid fully‑connected layer.
    The circuit contains a single qubit; the parameter theta controls
    an Ry rotation.  The class exposes a ``run`` method compatible
    with the classical counterpart and supports clipping and scaling
    of the parameters as in the fraud‑detection example.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100, backend: Optional[qiskit.providers.Backend] = None):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")

        # Build a simple parameterised circuit
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def clip_params(self, thetas: Iterable[float], bound: float = 5.0) -> np.ndarray:
        """Clip parameters to a symmetric interval."""
        return np.clip(np.asarray(thetas, dtype=float), -bound, bound)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit and return the expectation value.
        Parameters are clipped before binding to the circuit.
        """
        thetas = self.clip_params(thetas)
        param_binds = [{self.theta: theta} for theta in thetas]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()), dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])
