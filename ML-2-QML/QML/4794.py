"""Quantum module for the hybrid auto‑encoder.

Provides:
  * :class:`HybridQNN` – variational circuit that maps a classical latent vector
    to a quantum‑processed latent vector using a RealAmplitudes ansatz.
  * :class:`QuantumConv` – parameterised 2‑D filter inspired by the reference
    quantum convolution.
  * :class:`QuantumFCL` – simple one‑qubit parameterised circuit mimicking a fully‑connected layer.

The quantum circuit is built with Qiskit Aer and wrapped in a
:class:`qiskit_machine_learning.neural_networks.SamplerQNN` so that it can be
called from the classical network as a differentiable layer.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import AerSimulator
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN
import torch


# ----------------------------------------------------------------------
# Quantum latent processor
# ----------------------------------------------------------------------
class HybridQNN:
    """Variational quantum circuit that transforms a classical latent vector.

    The circuit consists of a RealAmplitudes ansatz followed by a full‑state
    measurement.  Parameters of the ansatz are set to the latent vector
    values, allowing the circuit to act as an implicit feature map.
    """

    def __init__(self, latent_dim: int, reps: int = 3, shots: int = 1024) -> None:
        self.latent_dim = latent_dim
        self.reps = reps
        self.shots = shots

        self.backend = AerSimulator()
        # Build the ansatz
        self.circuit = RealAmplitudes(latent_dim, reps=reps)
        self.circuit.measure_all()
        # SamplerQNN wraps the circuit and provides a tensor‑like API
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=latent_dim,
            sampler=self.backend,
        )

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to a batch of latent vectors.

        Args:
            latent: Tensor of shape (batch, latent_dim).

        Returns:
            Tensor of shape (batch, latent_dim) containing the expectation
            values produced by the sampler.
        """
        latent_np = latent.detach().cpu().numpy()
        # Bind each quantum parameter to the corresponding latent value
        param_binds = [
            {p: float(v) for p, v in zip(self.circuit.parameters, vec)}
            for vec in latent_np
        ]
        # Run the sampler
        result = self.qnn.run(input_data=[], parameter_binds=param_binds)
        # ``result`` is a numpy array of shape (batch, latent_dim)
        return torch.tensor(result, dtype=torch.float32, device=latent.device)


# ----------------------------------------------------------------------
# Quantum convolutional filter
# ----------------------------------------------------------------------
class QuantumConv:
    """Parameterized 2‑D filter that maps a kernel to a probability of |1>.

    The filter is a small circuit that applies RX rotations conditioned on the
    input pixel value and then executes a random Clifford circuit before
    measuring all qubits.  The output is the average probability of observing
    |1> across all qubits.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 512, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        self.backend = AerSimulator()
        self.circuit = qiskit.circuit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        # Add a random Clifford layer of depth 2
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the filter on a single kernel.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            Float between 0 and 1 representing the average |1> probability.
        """
        flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for vec in flat:
            bind = {}
            for i, val in enumerate(vec):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        ones = sum(int(key.count("1")) * val for key, val in counts.items())
        return ones / (total * self.n_qubits)


# ----------------------------------------------------------------------
# Quantum fully‑connected layer
# ----------------------------------------------------------------------
class QuantumFCL:
    """Tiny single‑qubit parameterised circuit that mimics a fully‑connected layer.

    It performs an H‑gate, a rotation about Y controlled by the parameter,
    and measures the qubit.  The expectation value of Z after the rotation
    serves as the output.
    """

    def __init__(self, shots: int = 256):
        self.backend = AerSimulator()
        self.shots = shots
        self.circuit = qiskit.circuit.QuantumCircuit(1)
        self.theta = Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameters.

        Args:
            thetas: 1‑D array of shape (batch,).

        Returns:
            1‑D array of shape (batch,) containing the expectation values.
        """
        param_binds = [{self.theta: float(t)} for t in thetas]
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectations = []
        for _ in thetas:
            key0 = "0"
            key1 = "1"
            cnt0 = counts.get(key0, 0)
            cnt1 = counts.get(key1, 0)
            exp = (cnt0 - cnt1) / self.shots
            expectations.append(exp)
        return np.array(expectations)


__all__ = ["HybridQNN", "QuantumConv", "QuantumFCL"]
