import math
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit

class QuanvCircuit:
    """
    Quantum quanvolution filter that mirrors the classical ConvFilter.
    It encodes input data as rotation angles and applies a random ansatz.
    """
    def __init__(self, kernel_size: int, backend, shots: int, threshold: int):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [QuantumCircuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # Data encoding
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()

        # Random feature circuit
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data):
        """
        Execute the quanvolution on classical data.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(self._circuit, self.backend,
                      shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

def Conv():
    """
    Factory that returns a QuanvCircuit instance with default settings.
    """
    backend = Aer.get_backend("qasm_simulator")
    return QuanvCircuit(kernel_size=2, backend=backend, shots=100, threshold=127)

def build_classifier_circuit(num_qubits: int, depth: int):
    """
    Construct a quantum classifier that first applies a quanvolution
    filter and then a layered variational ansatz of the specified depth.
    """
    # Ensure square qubit layout for the conv filter
    k = int(math.sqrt(num_qubits))
    if k * k!= num_qubits:
        raise ValueError("num_qubits must be a perfect square")

    conv = Conv()
    circuit = QuantumCircuit(num_qubits)

    # Append the quanvolution subcircuit
    circuit.append(conv._circuit.to_gate(), range(num_qubits))

    # Variational ansatz
    weights = ParameterVector("theta", num_qubits * depth)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    circuit.measure_all()

    encoding = list(conv.theta)
    weight_sizes = [num_qubits * depth]
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circuit, encoding, weight_sizes, observables

class QuantumConvClassifier:
    """
    Wrapper that exposes a unified interface for the quantum classifier.
    """
    def __init__(self, num_qubits: int, depth: int, backend=None, shots: int = 1024):
        self.circuit, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, data, params):
        """
        Execute the circuit with given data and variational parameters.

        Parameters
        ----------
        data : list[float]
            Data encoded into the first set of parameters (same length as num_qubits).
        params : list[float]
            Variational parameters for the ansatz (length = num_qubits * depth).

        Returns
        -------
        float
            Estimated probability of measuring |1> on the first qubit,
            used as the class score.
        """
        param_binds = {self.encoding[i]: data[i] for i in range(len(data))}
        for i, w in enumerate(params):
            param_binds[self.circuit.parameters[i]] = w

        job = execute(self.circuit, self.backend,
                      shots=self.shots, parameter_binds=[param_binds])
        result = job.result().get_counts(self.circuit)
        probs = {k: v / self.shots for k, v in result.items()}

        # Return probability of |1> on the first qubit (class 1)
        return probs.get("1" + "0" * (len(data) - 1), 0.0)

__all__ = ["build_classifier_circuit", "QuantumConvClassifier", "Conv"]
