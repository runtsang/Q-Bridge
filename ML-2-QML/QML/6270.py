import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit

def _build_classifier_circuit(num_qubits: int, depth: int):
    """
    Build a variational quantum circuit similar to a classical classifier.
    Adds a simple CZ connectivity pattern for entanglement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Encoding layer
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class ConvGen078:
    """
    Hybrid quantum convolution + classifier that mirrors the classical ConvGen078.
    Parameters
    ----------
    kernel_size : int
        Size of the 2‑D filter (default 2).
    threshold : int
        Threshold for the encoding (default 127).
    shots : int
        Number of shots for simulation (default 100).
    depth : int
        Depth of the variational circuit (default 1).
    backend : qiskit.providers.Backend
        Quantum backend; defaults to Aer qasm simulator.
    """

    def __init__(self, kernel_size: int = 2, threshold: int = 127,
                 shots: int = 100, depth: int = 1,
                 backend=None) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.circuit, self.encoding, self.weights, self.observables = _build_classifier_circuit(
            self.n_qubits, depth
        )
        self.circuit.measure_all()

    def run(self, data) -> float:
        """
        Execute the quantum circuit on a 2‑D input and return the average
        probability of measuring |1> across all qubits.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim!= 2 or data.shape!= (int(np.sqrt(self.n_qubits)), int(np.sqrt(self.n_qubits))):
            raise ValueError("Input must be a square 2‑D array matching kernel size.")
        flat = data.flatten()

        # Build parameter binding for the encoding layer
        param_bind = {}
        for i, val in enumerate(flat):
            param_bind[self.encoding[i]] = np.pi if val > self.threshold else 0.0

        # Bind variational parameters to zero for deterministic behaviour
        for w in self.weights:
            param_bind[w] = 0.0

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += sum(int(b) for b in bitstring) * freq

        prob_one = total_ones / (self.shots * self.n_qubits)
        return prob_one

__all__ = ["ConvGen078"]
