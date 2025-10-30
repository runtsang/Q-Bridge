"""Hybrid quantum circuit combining a parameterized rotation block
(simulating a fully connected layer) and an entangling block
(simulating self‑attention).  The interface mirrors the original
seeds: ``run`` accepts a sequence of theta values and an optional
input array (ignored)."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def FCL():
    class HybridQuantumLayer:
        def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
            self.n_qubits = n_qubits
            self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.circuit = QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            # Rotation block (fully connected analogue)
            for i in range(n_qubits):
                self.circuit.rx(self.theta, i)
                self.circuit.ry(self.theta, i)
                self.circuit.rz(self.theta, i)
            # Entangling block (self‑attention analogue)
            for i in range(n_qubits - 1):
                self.circuit.cx(i, i + 1)
            self.circuit.measure_all()

        def run(
            self,
            thetas: Iterable[float],
            inputs: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            # Bind one theta per qubit; if fewer provided, repeat last value
            param_binds = [{self.theta: theta} for theta in thetas]
            if len(param_binds) < self.n_qubits:
                param_binds += [
                    {self.theta: thetas[-1]} for _ in range(self.n_qubits - len(param_binds))
                ]
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array([int(k, 2) for k in result.keys()], dtype=float)
            probs = counts / self.shots
            expectation = np.sum(states * probs)
            return np.array([expectation])

    return HybridQuantumLayer()


__all__ = ["FCL"]
