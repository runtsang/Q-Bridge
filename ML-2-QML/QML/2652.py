"""Combined quantum convolution (quanvolution) and sampler QNN.

The ConvGen214 class first executes a parameterised quanvolution circuit
to extract a scalar feature from a 2‑D input.  The scalar is then fed
into a quantum sampler network that produces a probability distribution.
This mirrors the classical pipeline while leveraging quantum primitives.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN


def Conv():
    """Return a ConvGen214 instance for compatibility with the original API."""
    return ConvGen214()


class QuanvCircuit:
    """Quantum convolution (quanvolution) filter."""

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quanvolution circuit on a 2‑D array and return a scalar."""
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class SamplerQNNQuantum:
    """Quantum sampler network using a parameterised circuit."""

    def __init__(self) -> None:
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)

        self.sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def run(self, inputs: list[float]) -> np.ndarray:
        """Evaluate the sampler with the given input parameters."""
        bind = {self.inputs[0]: inputs[0], self.inputs[1]: inputs[1]}
        return self.sampler_qnn.run(bind)


class ConvGen214:
    """Hybrid quantum pipeline combining quanvolution and sampler QNN."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 100,
        backend=None,
    ) -> None:
        self.quanv = QuanvCircuit(
            kernel_size,
            backend or Aer.get_backend("qasm_simulator"),
            shots,
            threshold,
        )
        self.sampler_qnn = SamplerQNNQuantum()

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the full quantum pipeline and return the sampler output."""
        conv_value = self.quanv.run(data)
        # Feed the scalar twice to match the 2‑input sampler
        sampler_output = self.sampler_qnn.run([conv_value, conv_value])
        return sampler_output

    def run_quantum(self, data: np.ndarray) -> np.ndarray:
        """Convenience alias."""
        return self.run(data)


__all__ = ["ConvGen214", "Conv"]
