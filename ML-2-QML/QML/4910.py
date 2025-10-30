"""Hybrid quantum convolutional block (ConvGen86) – reference implementation."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute

# ----------------------------------------------------------------------
# Quantum sub‑modules (adapted from the seed files)
# ----------------------------------------------------------------------
def QuanvCircuit(kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127):
    class QuantumConv:
        def __init__(self) -> None:
            self.n_qubits = kernel_size ** 2
            self.circuit = QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i, t in enumerate(self.theta):
                self.circuit.rx(t, i)
            self.circuit.barrier()
            self.circuit += random_circuit(self.n_qubits, 2)
            self.circuit.measure_all()

            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for idx, val in enumerate(dat):
                    bind[self.theta[idx]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)

            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self.circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            return counts / (self.shots * self.n_qubits)

    return QuantumConv()


def QuantumSelfAttention(n_qubits: int = 4):
    class QuantumAttention:
        def __init__(self) -> None:
            self.n_qubits = n_qubits
            self.qr = QuantumCircuit(n_qubits)
            self.cr = QuantumCircuit(n_qubits)

        def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
            circuit = QuantumCircuit(self.n_qubits)
            for i in range(self.n_qubits):
                circuit.rx(rotation_params[3 * i], i)
                circuit.ry(rotation_params[3 * i + 1], i)
                circuit.rz(rotation_params[3 * i + 2], i)
            for i in range(self.n_qubits - 1):
                circuit.crx(entangle_params[i], i, i + 1)
            circuit.measure_all()
            return circuit

        def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> float:
            circuit = self._build_circuit(rotation_params, entangle_params)
            job = execute(circuit, backend, shots=shots)
            result = job.result().get_counts(circuit)
            # average number of |1> across all qubits
            total_ones = 0
            for key, val in result.items():
                ones = sum(int(b) for b in key)
                total_ones += ones * val
            return total_ones / (shots * self.n_qubits)

    return QuantumAttention()


def QuantumFCL(n_qubits: int = 1, backend=None, shots: int = 100):
    class QuantumFC:
        def __init__(self) -> None:
            self.circuit = QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self.circuit.h(range(n_qubits))
            self.circuit.barrier()
            self.circuit.ry(self.theta, range(n_qubits))
            self.circuit.measure_all()

            self.backend = backend
            self.shots = shots

        def run(self, thetas):
            job = execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array([int(k, 2) for k in result.keys()])
            probs = counts / self.shots
            expectation = np.sum(states * probs)
            return np.array([expectation])

    return QuantumFC()


def QuantumSamplerQNN():
    # Use a simple parameterised circuit and a statevector sampler
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = qiskit.primitives.StatevectorSampler()
    return sampler, qc, inputs, weights


# ----------------------------------------------------------------------
# Hybrid pipeline – the main class
# ----------------------------------------------------------------------
class ConvGen86:
    """
    Quantum‑enhanced hybrid convolutional network.

    Pipeline components:
    1. QuanvCircuit – parameterised quantum convolution.
    2. QuantumSelfAttention – attention block realised with a quantum circuit.
    3. QuantumFCL – single‑qubit fully‑connected layer.
    4. QuantumSamplerQNN – optional sampler that turns the final expectation
       into a probability distribution.
    """

    def __init__(self) -> None:
        backend = Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(2, backend, shots=100, threshold=127)
        self.attn = QuantumSelfAttention(4)
        self.fc = QuantumFCL(1, backend, shots=100)
        self.sampler, self.qc_sampler, self.inputs, self.weights = QuantumSamplerQNN()

    def run(
        self,
        data: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        thetas: Iterable[float],
        sampler_inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the quantum‑enhanced pipeline.

        Parameters
        ----------
        data : np.ndarray
            2‑D array for the quantum convolution.
        rotation_params, entangle_params : np.ndarray
            Parameters for the quantum attention circuit.
        thetas : Iterable[float]
            Parameters for the quantum fully‑connected layer.
        sampler_inputs : np.ndarray
            1‑D array of two parameters for the sampler circuit.

        Returns
        -------
        np.ndarray
            Probability vector produced by the sampler.
        """
        # 1. Quantum convolution
        conv_out = self.conv.run(data)

        # 2. Quantum self‑attention
        attn_out = self.attn.run(
            backend=Aer.get_backend("qasm_simulator"),
            rotation_params=rotation_params,
            entangle_params=entangle_params,
            shots=1024,
        )

        # 3. Quantum fully‑connected
        fc_out = self.fc.run(thetas).item()

        # 4. Combine signals into sampler parameters
        param_bind = {self.inputs[0]: sampler_inputs[0], self.inputs[1]: sampler_inputs[1],
                      self.weights[0]: conv_out, self.weights[1]: attn_out,
                      self.weights[2]: fc_out, self.weights[3]: 0.0}

        # Execute sampler circuit
        job = execute(self.qc_sampler, Aer.get_backend("statevector_simulator"), parameter_binds=[param_bind])
        statevector = job.result().get_statevector(self.qc_sampler)
        probs = np.abs(statevector) ** 2
        return probs

__all__ = ["ConvGen86"]
