"""Hybrid quantum layer that combines a single‑qubit fully connected circuit
with a 2×2 quanvolution filter.  The two expectation values are summed
with a tunable weight ``alpha``.
"""

import numpy as np
from typing import Iterable

import qiskit
from qiskit.circuit.random import random_circuit


class HybridQuantumCircuit:
    """Quantum equivalent of :class:`HybridLayer`."""

    def __init__(
        self,
        n_qubits: int = 1,
        kernel_size: int = 2,
        threshold: float = 127,
        backend=None,
        shots: int = 100,
        alpha: float = 0.5,
    ) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.alpha = alpha

        # Fully connected part – single qubit circuit
        self.fc_circuit = qiskit.QuantumCircuit(n_qubits)
        self.fc_theta = qiskit.circuit.Parameter("theta")
        self.fc_circuit.h(range(n_qubits))
        self.fc_circuit.barrier()
        self.fc_circuit.ry(self.fc_theta, range(n_qubits))
        self.fc_circuit.measure_all()

        # Quanvolution part – 2×2 filter = 4 qubits
        self.n_qubits_q = kernel_size ** 2
        self.qc = qiskit.QuantumCircuit(self.n_qubits_q)
        self.theta_q = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits_q)
        ]
        for i in range(self.n_qubits_q):
            self.qc.rx(self.theta_q[i], i)
        self.qc.barrier()
        self.qc += random_circuit(self.n_qubits_q, 2)
        self.qc.measure_all()
        self.threshold = threshold

    def _expectation(self, circuit: qiskit.QuantumCircuit, params: dict) -> float:
        job = qiskit.execute(
            circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[params],
        )
        result = job.result().get_counts(circuit)
        # Expectation of measuring |1> on first qubit (or average over all)
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        states = np.array([int(k) for k in result.keys()]).astype(float)
        return np.sum(states * probs)

    def run(self, thetas: Iterable[float], data: np.ndarray) -> np.ndarray:
        """Run both sub‑circuits and return a weighted sum of their
        expectation values.
        """
        # Fully connected expectation
        fc_params = {self.fc_theta: thetas[0]}
        fc_exp = self._expectation(self.fc_circuit, fc_params)

        # Quanvolution expectation
        bind = {}
        flat = data.reshape(-1)
        for i, val in enumerate(flat):
            bind[self.theta_q[i]] = np.pi if val > self.threshold else 0
        qc_exp = self._expectation(self.qc, bind)

        combined = self.alpha * fc_exp + (1 - self.alpha) * qc_exp
        return np.array([combined])


def FCL() -> HybridQuantumCircuit:
    """Return an instance of the hybrid quantum circuit."""
    return HybridQuantumCircuit()


__all__ = ["HybridQuantumCircuit", "FCL"]
