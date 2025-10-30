import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a quantum ansatz identical to the one used in the
    reference `QuantumClassifierModel.py`, but returning the circuit
    and its parameter metadata for use by the wrapper class.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
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

class QuantumHybridCircuit:
    """
    Variational quantum circuit that accepts a parameter vector
    produced by the classical network.  The circuit is built from
    an encoding layer (Rx rotations) followed by `depth` blocks of
    Ry rotations and nearest‑neighbour CZ gates.  The expectation
    value of a set of Z observables is returned, matching the
    behaviour of the original `build_classifier_circuit`.
    """
    def __init__(self, n_qubits: int, depth: int,
                 backend=None, shots: int = 1024) -> None:
        self.circuit, self.encoding, self.weights, self.observables = \
            build_classifier_circuit(n_qubits, depth)
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, theta: Iterable[float]) -> np.ndarray:
        """
        Bind the classical parameters to the circuit and execute.
        `theta` must contain `len(self.weights)` values.
        """
        param_dict = {self.weights[i]: theta[i] for i in range(len(self.weights))}
        bound_circuit = self.circuit.bind_parameters(param_dict)
        job = qiskit.execute(
            bound_circuit, self.backend, shots=self.shots
        )
        result = job.result()
        counts = result.get_counts(bound_circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])
