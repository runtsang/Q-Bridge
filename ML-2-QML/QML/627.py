"""ConvFilter: quantum convolution filter with parameterized ansatz and gradient estimation.

This class implements a quantum convolution filter that can be used as a drop‑in replacement for the classical Conv filter. It supports a variational ansatz, expectation‑value readout of Pauli‑Z, parameter‑shift gradient estimation, and configurable backend, shots and threshold.

Example:

    from ConvFilter import ConvFilter
    filter = ConvFilter(kernel_size=2, backend="qasm_simulator",
                        shots=1024, threshold=0.5)
    out = filter.run(np.random.rand(2,2))
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Sequence, Optional, Callable

__all__ = ["ConvFilter"]


class ConvFilter:
    """
    Quantum convolution filter.

    Parameters
    ----------
    kernel_size : int
        Size of the filter (kernel) – the filter will act on a
        ``kernel_size x kernel_size`` patch.
    backend : str or qiskit.providers.Backend, optional
        Backend used for execution.  If ``None`` a local Aer
        simulator is used.
    shots : int
        Number of shots for each evaluation.
    threshold : float
        Threshold applied to the classical input before mapping to
        rotation angles.  Values greater than threshold are mapped to
        ``π`` otherwise ``0``.
    ansatz_depth : int, optional
        Number of variational layers in the circuit.
    optimizer : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        Function that updates the parameters during training.  It
        receives the current parameters and the gradient and returns
        the updated parameters.
    """

    def __init__(
        self,
        kernel_size: int,
        backend: Optional[str | qiskit.providers.Backend] = None,
        shots: int = 1024,
        threshold: float = 0.5,
        ansatz_depth: int = 2,
        optimizer: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.optimizer = optimizer

        # Resolve backend
        if backend is None:
            self.backend = Aer.get_backend("qasm_simulator")
        elif isinstance(backend, str):
            self.backend = Aer.get_backend(backend)
        else:
            self.backend = backend

        # Parameter vector
        self.theta = np.random.uniform(0, 2 * np.pi, size=self.n_qubits * ansatz_depth)

        # Build circuit template
        self.circuit = self._build_ansatz(ansatz_depth)

    def _build_ansatz(self, depth: int) -> QuantumCircuit:
        """Create a parameterized ansatz with rotation layers and entangling CNOTs."""
        circuit = QuantumCircuit(self.n_qubits)
        params = [Parameter(f"θ_{i}") for i in range(self.n_qubits * depth)]

        for d in range(depth):
            start = d * self.n_qubits
            for q in range(self.n_qubits):
                circuit.rx(params[start + q], q)

            # Entangle with a simple nearest‑neighbour pattern
            for q in range(self.n_qubits - 1):
                circuit.cx(q, q + 1)
            # Optional wrap‑around entanglement
            circuit.cx(self.n_qubits - 1, 0)

        circuit.measure_all()
        return circuit

    def _data_to_params(self, data: np.ndarray) -> Sequence[float]:
        """Map classical data to rotation angles using the threshold."""
        flattened = data.flatten()
        angles = np.where(flattened > self.threshold, np.pi, 0.0)
        return angles

    def run(self, data: np.ndarray) -> float:
        """
        Execute the filter on a single kernel patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            Average probability of measuring ``|1⟩`` across all qubits.
        """
        # Prepare parameter binding
        angle_bind = dict(zip(self.circuit.parameters[:self.n_qubits], self._data_to_params(data)))
        var_bind = dict(zip(self.circuit.parameters[self.n_qubits:], self.theta))
        bind = {**angle_bind, **var_bind}

        # Execute
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average probability of measuring |1> per qubit
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq

        return total_ones / (self.shots * self.n_qubits)

    def expectation_z(self, data: np.ndarray) -> np.ndarray:
        """
        Return expectation values of Pauli‑Z on each qubit.

        Parameters
        ----------
        data : np.ndarray
            Input patch.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_qubits,)`` containing ⟨Z⟩ for each qubit.
        """
        angle_bind = dict(zip(self.circuit.parameters[:self.n_qubits], self._data_to_params(data)))
        var_bind = dict(zip(self.circuit.parameters[self.n_qubits:], self.theta))
        bind = {**angle_bind, **var_bind}

        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)

        exp_z = np.zeros(self.n_qubits)
        for bitstring, freq in counts.items():
            for i, bit in enumerate(bitstring[::-1]):  # LSB first
                exp_z[i] += (1 if bit == "0" else -1) * freq
        exp_z /= self.shots
        return exp_z

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Estimate gradient of the output with respect to the variational
        parameters using the parameter‑shift rule.

        Parameters
        ----------
        data : np.ndarray
            Input patch.

        Returns
        -------
        np.ndarray
            Gradient vector of shape ``(n_params,)``.
        """
        grad = np.zeros_like(self.theta)
        shift = np.pi / 2

        for idx in range(len(self.theta)):
            theta_plus = self.theta.copy()
            theta_minus = self.theta.copy()
            theta_plus[idx] += shift
            theta_minus[idx] -= shift

            bind_plus = dict(zip(self.circuit.parameters[self.n_qubits:], theta_plus))
            bind_minus = dict(zip(self.circuit.parameters[self.n_qubits:], theta_minus))
            angle_bind = dict(zip(self.circuit.parameters[:self.n_qubits], self._data_to_params(data)))

            job_plus = execute(self.circuit, self.backend,
                               shots=self.shots,
                               parameter_binds=[{**angle_bind, **bind_plus}])
            job_minus = execute(self.circuit, self.backend,
                                shots=self.shots,
                                parameter_binds=[{**angle_bind, **bind_minus}])

            out_plus = job_plus.result().get_counts(self.circuit)
            out_minus = job_minus.result().get_counts(self.circuit)

            def prob1(counts):
                total = 0
                for bitstring, freq in counts.items():
                    total += bitstring.count("1") * freq
                return total / (self.shots * self.n_qubits)

            grad[idx] = (prob1(out_plus) - prob1(out_minus)) / 2

        return grad

    def train_step(self, data: np.ndarray, lr: float = 0.01) -> None:
        """
        Perform a single gradient‑descent update on the variational
        parameters.

        Parameters
        ----------
        data : np.ndarray
            Input patch to evaluate.
        lr : float
            Learning rate.
        """
        grad = self.gradient(data)
        self.theta -= lr * grad
