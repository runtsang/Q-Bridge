"""Hybrid quantum fully‑connected layer with a kernel‑style data encoding.

The quantum implementation mirrors the classical ``HybridFCLKernel`` but
uses a parameterised quantum circuit.  The circuit consists of a
kernel‑encoding stage (Ry rotations on each qubit) followed by a
variational rotation that acts as the linear read‑out.  The ``run`` method
accepts a list of parameters ``thetas`` where the first element is the
variational angle and the remaining elements encode the input data.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from typing import Iterable

class HybridFCLKernel:
    """
    Quantum analogue of :class:`HybridFCLKernel` from the classical
    implementation.  The circuit encodes the input data via Ry gates
    and applies a single variational Rz rotation on the first qubit.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        backend=None,
        shots: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        n_qubits
            Number of qubits used for the kernel encoding.
        backend
            Qiskit backend.  If ``None`` a local qasm simulator is used.
        shots
            Number of shots for circuit execution.
        """
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct the parameterised circuit."""
        self.circuit = QuantumCircuit(self.n_qubits)
        # Parameters for kernel encoding
        self.kernel_params = [
            qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self.circuit.ry(self.kernel_params[i], i)
        # Variational read‑out rotation
        self.v_param = qiskit.circuit.Parameter("v")
        self.circuit.rz(self.v_param, 0)
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float], x=None) -> np.ndarray:
        """
        Execute the circuit and return the expectation value of Z on the
        first qubit.

        Parameters
        ----------
        thetas
            Iterable of length ``n_qubits + 1``.  ``thetas[0]`` is the
            variational angle ``v`` and the rest are the kernel angles
            that encode the input data.
        x
            Unused; present for API compatibility with the classical
            version.

        Returns
        -------
        np.ndarray
            Output of shape ``(1,)`` containing the expectation value.
        """
        if len(thetas)!= self.n_qubits + 1:
            raise ValueError(
                f"Expected {self.n_qubits + 1} theta values, got {len(thetas)}."
            )
        v = thetas[0]
        kernel_thetas = thetas[1:]
        param_bind = {self.v_param: v}
        for i, theta in enumerate(kernel_thetas):
            param_bind[self.kernel_params[i]] = theta
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(state, 2) for state in counts.keys()])
        # Expectation of Z on qubit 0: +1 for |0>, -1 for |1>
        exp = np.sum((1 - 2 * (states & 1)) * probs)
        return np.array([exp])

__all__ = ["HybridFCLKernel"]
