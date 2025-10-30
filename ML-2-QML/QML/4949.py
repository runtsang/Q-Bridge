"""Hybrid quantum network that mirrors the classical HybridNet.

The class ``HybridNet`` contains quantum‑parameterized circuits for:
    * Fully‑connected layer (1‑qubit Ry rotation).
    * Quanvolution filter (2×2 patch encoded on 4 qubits with a random circuit).
    * SamplerQNN (parameterized 2‑qubit circuit with a state‑vector sampler).

The API matches the classical counterpart: a ``run`` method that
accepts the same inputs and returns expectation values or probabilities.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

__all__ = ["HybridNet"]


def FCL(backend: qiskit.providers.Provider | None = None, shots: int = 1024):
    """Quantum fully‑connected layer (1‑qubit variational circuit)."""
    class QuantumFCL:
        def __init__(self) -> None:
            self.backend = backend or Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.theta = Parameter("theta")
            qc = qiskit.QuantumCircuit(1)
            qc.h(0)
            qc.ry(self.theta, 0)
            qc.measure_all()
            self.circuit = qc

        def run(self, theta_value: float) -> np.ndarray:
            job = execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta_value}],
            )
            result = job.result().get_counts(self.circuit)
            probs = np.array([int(k, 2) for k in result.keys()], dtype=float)
            counts = np.array(list(result.values()))
            expectation = (probs * counts).sum() / self.shots
            return np.array([expectation])

    return QuantumFCL()


def Conv(
    kernel_size: int = 2,
    threshold: float = 127,
    backend: qiskit.providers.Provider | None = None,
    shots: int = 1024,
):
    """Quantum quanvolution filter for 2×2 patches (4 qubits)."""
    class QuantumConv:
        def __init__(self) -> None:
            self.n_qubits = kernel_size ** 2
            self.threshold = threshold
            self.backend = backend or Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
            qc = qiskit.QuantumCircuit(self.n_qubits)
            for i in range(self.n_qubits):
                qc.rx(self.theta[i], i)
            qc.barrier()
            qc += qiskit.circuit.random.random_circuit(self.n_qubits, depth=2)
            qc.measure_all()
            self.circuit = qc

        def run(self, data: np.ndarray) -> float:
            """
            Parameters
            ----------
            data
                2‑D array of shape (kernel_size, kernel_size) with integer pixel values.
            Returns
            -------
            float
                Average probability of measuring |1> over all qubits.
            """
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for row in data:
                bind = {}
                for i, val in enumerate(row):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)
            job = execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self.circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(b) for b in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)

    return QuantumConv()


def SamplerQNN(
    backend: qiskit.providers.Provider | None = None,
    shots: int = 1024,
):
    """Quantum sampler network (2‑qubit variational circuit)."""
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 4)

    qc2 = qiskit.QuantumCircuit(2)
    qc2.ry(inputs2[0], 0)
    qc2.ry(inputs2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[0], 0)
    qc2.ry(weights2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[2], 0)
    qc2.ry(weights2[3], 1)

    sampler = Sampler()
    sampler_qnn = QSamplerQNN(
        circuit=qc2,
        input_params=inputs2,
        weight_params=weights2,
        sampler=sampler,
    )
    return sampler_qnn


class HybridNet:
    """Hybrid quantum network mirroring the classical HybridNet API."""
    def __init__(
        self,
        backend: qiskit.providers.Provider | None = None,
        shots: int = 1024,
    ) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.fcl = FCL(backend=self.backend, shots=self.shots)
        self.conv = Conv(backend=self.backend, shots=self.shots)
        self.sampler = SamplerQNN(backend=self.backend, shots=self.shots)

    def run(
        self,
        theta_value: float,
        patch: np.ndarray,
        sampler_input: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Execute all quantum sub‑modules with the provided inputs.

        Parameters
        ----------
        theta_value
            Rotation angle for the FCL circuit.
        patch
            2×2 image patch for the quanvolution circuit.
        sampler_input
            1‑D array of length 2 for the SamplerQNN.

        Returns
        -------
        tuple
            (fcl_output, conv_output, sampler_output)
        """
        fcl_out = self.fcl.run(theta_value)
        conv_out = self.conv.run(patch)
        # SamplerQNN expects a 2‑D array: (batch, 2).  We provide a batch of 1.
        sampler_out = self.sampler.predict(
            np.array([sampler_input], dtype=np.float32)
        )
        return fcl_out, conv_out, sampler_out
