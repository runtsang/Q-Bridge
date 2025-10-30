"""
Quantum convolution module using a variational ansatz and a trainable threshold.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance


class Conv:
    """
    A quantum filter that applies a variational circuit to a flattened kernel.
    The circuit contains parameterized rotations and entangling layers.
    The output is the expectation value of Z on the first qubit.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 127.0,
        entanglement: str = "linear",
        reps: int = 2,
    ) -> None:
        """
        Args:
            kernel_size: Size of the square kernel (k x k).
            shots: Number of shots for simulation.
            threshold: Classical threshold for binarizing input.
            entanglement: Entanglement pattern for the ansatz.
            reps: Number of repetitions in the ansatz.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.entanglement = entanglement
        self.reps = reps

        # Create a parameterized ansatz
        self.params = [Parameter(f"θ{i}") for i in range(self.n_qubits * reps * 3)]
        self.circuit = RealAmplitudes(
            self.n_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=True,
        )
        self.circuit.name = "ConvAnsatz"

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")
        self.qi = QuantumInstance(self.backend, shots=self.shots, seed_simulator=42)

    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        """
        Flatten and binarize the kernel data according to the threshold.
        """
        flat = data.flatten()
        binarized = np.where(flat > self.threshold, np.pi, 0.0)
        return binarized

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on the input kernel and return the
        expectation value of Z on the first qubit.

        Args:
            data: 2D array of shape (kernel_size, kernel_size).

        Returns:
            Expectation value in [-1, 1].
        """
        bin_data = self._prepare_data(data)
        param_binds = [{p: val for p, val in zip(self.params, bin_data)}]

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts()
        # Compute expectation of Z on qubit 0
        exp = 0.0
        for bitstring, cnt in counts.items():
            if bitstring[-1] == "0":
                exp += cnt
            else:
                exp -= cnt
        exp /= self.shots
        return exp

    def train(self, data_loader, optimizer, loss_fn, epochs: int = 10):
        """
        Placeholder training loop for the variational circuit.
        In practice, one would use a quantum-aware optimizer or gradient
        estimation method. Here we provide a simple classical gradient
        estimation using finite differences.

        Args:
            data_loader: Iterable yielding (input, target) pairs.
            optimizer: Optimizer that updates self.params.
            loss_fn: Loss function.
            epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            for x, y in data_loader:
                # Estimate gradient via finite differences
                grads = np.zeros_like(self.params)
                eps = 1e-3
                base = self.run(x)
                for i, p in enumerate(self.params):
                    perturbed_params = np.array([eps if j == i else 0.0 for j in range(len(self.params))])
                    val = self.run(x)  # In a real implementation we would bind perturbed_params
                    grads[i] = (val - base) / eps
                # Update parameters
                for i, p in enumerate(self.params):
                    p_val = float(p)
                    p_val -= optimizer.learning_rate * grads[i]
                    p_val = max(min(p_val, np.pi), -np.pi)
                    p._value = p_val
