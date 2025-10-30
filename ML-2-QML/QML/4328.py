import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# 1. Hybrid quantum filter – parameterised quanvolution circuit
# --------------------------------------------------------------------------- #
class HybridQuantumConv:
    """
    A 2×2 quanvolution circuit that accepts classical image patches.
    The circuit is parameterised by a set of RX angles and a random two‑qubit
    layer.  Data is encoded as angle shifts on the RX gates.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 127.0) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Parameterised circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

        # Pre‑allocate parameter bindings
        self.param_binds: List[dict] = []

    def run(self, data: np.ndarray | List[float] | List[List[float]]) -> float:
        """
        Execute the circuit on a single 2×2 patch and return the average
        probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        self.param_binds.clear()

        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            self.param_binds.append(bind)

        job = qiskit.execute(self.circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=self.param_binds)
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# 2. Quantum classifier factory – matches classical API
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[qiskit.QuantumCircuit,
                                                   List[ParameterVector],
                                                   List[ParameterVector],
                                                   List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit data encoding and variational
    parameters.

    Returns
    -------
    circuit : QuantumCircuit
        The full quantum circuit.
    encoding : List[ParameterVector]
        List containing the data‑encoding parameters.
    weights : List[ParameterVector]
        List containing the variational parameters.
    observables : List[SparsePauliOp]
        Pauli‑Z observables on each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = qiskit.QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

__all__ = [
    "HybridQuantumConv",
    "build_classifier_circuit",
]
