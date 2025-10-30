import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit

class HybridFCL:
    """
    Quantum implementation of the hybrid fully‑connected layer.

    Supports three modes that mirror the classical variants:

    * ``classical`` – a single‑qubit Ry ansatz that reproduces the
      original FCL circuit.
    * ``kernel`` – a fixed ansatz that implements an RBF‑style kernel
      through a sequence of Ry and CZ gates.
    * ``conv`` – a quanvolution‑style filter that applies a random
      circuit to a small patch of qubits.

    The constructor arguments are intentionally compatible with the
    classical ``HybridFCL`` so that the API is drop‑in.
    """

    def __init__(self,
                 n_qubits: int = 1,
                 mode: str = "classical",
                 depth: int = 1,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 127):
        self.mode = mode
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        if mode == "classical":
            self.circuit = QuantumCircuit(n_qubits)
            theta = ParameterVector("theta", length=n_qubits)
            self.circuit.h(range(n_qubits))
            self.circuit.ry(theta, range(n_qubits))
            self.circuit.measure_all()

        elif mode == "kernel":
            self.circuit = QuantumCircuit(n_qubits)
            enc = ParameterVector("x", length=n_qubits)
            var = ParameterVector("theta", length=n_qubits * depth)
            self.circuit.rx(enc, range(n_qubits))
            idx = 0
            for _ in range(depth):
                for q in range(n_qubits):
                    self.circuit.ry(var[idx], q)
                    idx += 1
                for q in range(n_qubits - 1):
                    self.circuit.cz(q, q + 1)
            self.circuit.measure_all()
            self.enc = enc
            self.var = var

        elif mode == "conv":
            # each pixel of a kernel is mapped to a qubit
            self.circuit = QuantumCircuit(n_qubits * n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.circuit.num_qubits)]
            for i in range(self.circuit.num_qubits):
                self.circuit.rx(self.theta[i], i)
            self.circuit.barrier()
            self.circuit += random_circuit(self.circuit.num_qubits, 2)
            self.circuit.measure_all()
            self.threshold = threshold

        else:
            raise ValueError(f"unknown mode {mode}")

    def run(self, thetas):
        if self.mode == "classical":
            param_binds = [{self.circuit.parameters[i]: theta} for i, theta in enumerate(thetas)]
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            probs = counts / self.shots
            states = np.array(list(result.keys())).astype(float)
            expectation = np.sum(states * probs)
            return np.array([expectation])

        if self.mode == "kernel":
            # bind input data to encoding parameters
            param_binds = [{self.enc[i]: theta} for i, theta in enumerate(thetas)]
            # fix variational parameters to zero
            for i in range(len(self.var)):
                param_binds[0][self.var[i]] = 0.0
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            probs = counts / self.shots
            states = np.array(list(result.keys())).astype(float)
            expectation = np.sum(states * probs)
            return np.array([expectation])

        if self.mode == "conv":
            # bind data as theta values according to a threshold
            param_binds = []
            for data in thetas:
                bind = {}
                for i, val in enumerate(data):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
                param_binds.append(bind)

            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self.circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            expectation = counts / (self.shots * self.circuit.num_qubits)
            return np.array([expectation])

        raise RuntimeError("unreachable")

__all__ = ["HybridFCL"]
