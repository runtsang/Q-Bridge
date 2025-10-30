"""Variational quantum circuit mimicking a fully‑connected layer with entanglement and gradient estimation."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter


class FullyConnectedLayer:
    """
    A parameterized quantum circuit that simulates a fully‑connected layer.
    The circuit applies a trainable Ry rotation to each qubit, entangles them
    via a linear CNOT chain, and measures all qubits.  The returned expectation
    value is the weighted sum of computational basis states.  A simple
    parameter‑shift rule is provided for gradient estimation.
    """

    def __init__(self, n_qubits: int = 4, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameter to be bound per qubit
        self.theta = Parameter("theta")

        # Build the circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        for q in range(n_qubits):
            self.circuit.ry(self.theta, q)
        for q in range(n_qubits - 1):
            self.circuit.cx(q, q + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameter vector.  ``thetas`` must
        have length equal to ``n_qubits``.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {len(thetas)}"
            )
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

    def parameter_shift_gradients(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Estimate gradients of the expectation value w.r.t. each parameter using
        the parameter‑shift rule: g = (f(θ+π/2) - f(θ-π/2)) / 2.
        """
        grads = []
        for i in range(self.n_qubits):
            shift = np.pi / 2
            thetas_plus = list(thetas)
            thetas_minus = list(thetas)
            thetas_plus[i] += shift
            thetas_minus[i] -= shift
            f_plus = self.run(thetas_plus)[0]
            f_minus = self.run(thetas_minus)[0]
            grads.append((f_plus - f_minus) / 2)
        return np.array(grads)


__all__ = ["FullyConnectedLayer"]
