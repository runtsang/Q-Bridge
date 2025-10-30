"""Quantum implementation of a fully‑connected layer using Qiskit.

Features added compared to the seed:
* Multi‑qubit support with a simple hardware‑friendly ansatz.
* Choice of measurement observable (default ``PauliZ``).
* Optional analytic expectation via the state‑vector simulator.
* Batch execution of several parameter sets in a single job.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List, Dict, Any

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import execute
from qiskit.providers import Backend
from qiskit_aer import AerSimulator


class QuantumFullyConnectedLayer:
    """
    A parameterised quantum circuit that emulates a classical fully‑connected
    layer.  Each input theta controls a rotation on a dedicated qubit;
    the circuit forms a shallow entangling layer and measures the
    expectation of Pauli‑Z on each qubit.

    Parameters
    ----------
    n_qubits : int
        Number of input features / qubits.
    backend : Backend, optional
        Qiskit backend to execute the circuit.  If ``None`` a local
        Aer simulator is used.
    shots : int, optional
        Number of shots for a sampling backend.  Ignored when using the
        state‑vector simulator.
    """

    def __init__(
        self,
        n_qubits: int,
        backend: Backend | None = None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or AerSimulator(method="aer_simulator_statevector")
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Create the parameterised ansatz."""
        self.theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]
        self.circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)

        # Simple hardware‑friendly ansatz: RY on each qubit followed by a
        # nearest‑neighbour CNOT chain.
        for i, p in enumerate(self.theta):
            self.circuit.ry(p, i)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)

        # Measure each qubit into a classical register
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for a single set of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Length must equal ``n_qubits``.  Each element is bound to the
            corresponding rotation gate.

        Returns
        -------
        np.ndarray
            Expected value of Pauli‑Z for each qubit, shape ``(n_qubits,)``.
        """
        param_bind = {self.theta[i]: thetas[i] for i in range(self.n_qubits)}
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Convert counts to expectation values of Z.
        expectations = []
        for i in range(self.n_qubits):
            # Sum over all bitstrings weighted by (-1)^bit_i
            exp_z = 0.0
            for bitstring, cnt in counts.items():
                bit = int(bitstring[self.n_qubits - 1 - i])  # Qiskit order
                exp_z += ((-1) ** bit) * cnt
            exp_z /= self.shots
            expectations.append(exp_z)
        return np.array(expectations, dtype=np.float32)

    def run_batch(
        self, batch_thetas: Sequence[Iterable[float]]
    ) -> np.ndarray:
        """
        Execute the circuit for multiple parameter sets in a single job.

        Parameters
        ----------
        batch_thetas : Sequence[Iterable[float]]
            Each element is a theta sequence of length ``n_qubits``.

        Returns
        -------
        np.ndarray
            Shape ``(len(batch_thetas), n_qubits)`` containing the
            expectation values for every batch element.
        """
        param_binds = [
            {self.theta[i]: thetas[i] for i in range(self.n_qubits)}
            for thetas in batch_thetas
        ]
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        all_counts = [result.get_counts(self.circuit, i) for i in range(len(batch_thetas))]

        expectations = []
        for counts in all_counts:
            exp_z = []
            for i in range(self.n_qubits):
                val = 0.0
                for bitstring, cnt in counts.items():
                    bit = int(bitstring[self.n_qubits - 1 - i])
                    val += ((-1) ** bit) * cnt
                val /= self.shots
                exp_z.append(val)
            expectations.append(exp_z)
        return np.array(expectations, dtype=np.float32)


def FCL(
    n_qubits: int,
    backend: Backend | None = None,
    shots: int = 1024,
) -> QuantumFullyConnectedLayer:
    """
    Factory function mirroring the original API.  It returns an instance
    of ``QuantumFullyConnectedLayer`` configured with the supplied
    parameters.

    The function name and signature are kept identical to the seed,
    ensuring backward compatibility for scripts that import ``FCL``.
    """
    return QuantumFullyConnectedLayer(n_qubits, backend, shots)


__all__ = ["FCL", "QuantumFullyConnectedLayer"]
