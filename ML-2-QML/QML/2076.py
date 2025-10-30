"""Quantum implementation of a fully‑connected layer using a parameterised ansatz.

The circuit is a stack of alternating RY and RZ rotations followed by CNOT
entanglement. A single measurement in the computational basis yields a
bit‑string that is interpreted as a real number; its expectation value is
returned. Parameters are supplied as an iterable of floats and are bound
batch‑wise to the circuit for efficient evaluation on a simulator.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from typing import Iterable

class FullyConnectedLayer:
    def __init__(
        self,
        n_qubits: int = 2,
        depth: int = 3,
        backend=None,
        shots: int = 1024,
    ) -> None:
        """Create a parameterised quantum circuit.

        Parameters
        ----------
        n_qubits: int
            Number of qubits in the ansatz.
        depth: int
            Number of alternating RY/RZ layers.
        backend: qiskit.providers.provider.Provider
            Backend to execute the circuit on; defaults to the local Aer
            simulator.
        shots: int
            Number of shots per execution.
        """
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = self._build_circuit(n_qubits, depth)

    def _build_circuit(self, n_qubits: int, depth: int) -> QuantumCircuit:
        circ = QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        # Entangle the qubits with a simple CNOT ladder
        for i in range(n_qubits - 1):
            circ.cx(i, i + 1)
        # Parameterised layers
        for _ in range(depth):
            circ.ry(theta, range(n_qubits))
            circ.rz(theta, range(n_qubits))
            circ.cx(0, 1)  # one entangling gate per layer
        circ.measure_all()
        return circ

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a batch of parameters.

        Each theta is bound to the single parameter ``theta`` in the circuit.
        The measurement outcomes are converted to floating‑point values by
        interpreting the bit‑strings as binary numbers. The expectation value
        is the weighted average over all shots.
        """
        # Bind each theta to a separate circuit execution
        params = [{self.circuit.parameters[0]: t} for t in thetas]
        jobs = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=params,
        )
        result = jobs.result()
        expectation = []
        for theta in thetas:
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            expectation.append(np.sum(states * probs))
        return np.array(expectation)
