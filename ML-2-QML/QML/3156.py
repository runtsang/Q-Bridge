"""Hybrid fully connected + convolutional layer with quantum back‑end."""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer

__all__ = ["HybridQuantumCircuit", "FCL"]


class HybridQuantumCircuit:
    """
    Quantum implementation that combines a single‑qubit parameterised
    circuit (linear) with a multi‑qubit quanvolution (convolution).

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the convolutional part (kernel_size ** 2).
    kernel_size : int
        Size of the convolutional kernel.
    backend : qiskit.providers.BaseBackend
        Backend to execute the circuits on.
    shots : int
        Number of shots for expectation estimation.
    threshold : float
        Threshold used to map classical pixel values to rotation angles.
    """

    def __init__(
        self,
        n_qubits: int,
        kernel_size: int,
        backend,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        # --- Linear branch (single qubit) ---
        self.linear_circ = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.linear_circ.h(0)
        self.linear_circ.barrier()
        self.linear_circ.ry(self.theta, 0)
        self.linear_circ.measure_all()

        # --- Convolutional branch ---
        self.n_qubits = n_qubits
        self.conv_circ = qiskit.QuantumCircuit(n_qubits)
        self.theta_params = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)
        ]
        for i in range(n_qubits):
            self.conv_circ.rx(self.theta_params[i], i)
        self.conv_circ.barrier()
        self.conv_circ += random_circuit(n_qubits, depth=2)
        self.conv_circ.measure_all()

    def _run_linear(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the linear sub‑circuit."""
        job = execute(
            self.linear_circ,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.linear_circ)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def _run_conv(self, data: Union[Sequence[Sequence[float]], np.ndarray]) -> float:
        """Execute the convolutional sub‑circuit."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in data:
            bind = {}
            for i, val in enumerate(row):
                bind[self.theta_params[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self.conv_circ,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.conv_circ)

        total_ones = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total_ones += ones * val

        return total_ones / (self.shots * self.n_qubits)

    def run(self, inputs: Union[Iterable[float], Sequence[Sequence[float]]]) -> Union[np.ndarray, float]:
        """
        Dispatch to the appropriate quantum sub‑circuit.

        Returns
        -------
        np.ndarray or float
            Linear expectation for iterable inputs; convolutional
            mean‑ones probability for 2‑D array inputs.
        """
        if isinstance(inputs, (list, tuple, np.ndarray)):
            if all(isinstance(x, (int, float)) for x in inputs):
                return self._run_linear(inputs)
            else:
                return self._run_conv(inputs)
        raise TypeError("Unsupported input type for HybridQuantumCircuit.run")

    def __call__(self, inputs: Union[Iterable[float], Sequence[Sequence[float]]]) -> Union[np.ndarray, float]:
        return self.run(inputs)


def FCL() -> HybridQuantumCircuit:
    """
    Factory function mirroring the original API.

    Returns
    -------
    HybridQuantumCircuit
        An instance ready for use with either linear or convolutional data.
    """
    simulator = Aer.get_backend("qasm_simulator")
    # Example: 2x2 kernel → 4 qubits for convolution
    return HybridQuantumCircuit(n_qubits=4, kernel_size=2, backend=simulator, shots=100, threshold=127)
