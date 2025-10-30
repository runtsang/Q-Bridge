import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, QNN
from qiskit.primitives import StatevectorSampler

def build_classifier_circuit(num_qubits: int,
                            depth: int) -> Tuple[QuantumCircuit,
                                                 Iterable[ParameterVector],
                                                 Iterable[ParameterVector],
                                                 list[SparsePauliOp]]:
    """
    Constructs a layered ansatz with explicit encoding and variational parameters.
    Mirrors the classical feed‑forward architecture: each depth adds a set of
    single‑qubit rotations followed by a nearest‑neighbour CZ coupling.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
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

    # Observables for a simple binary read‑out
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class HybridFCL:
    """
    Quantum analogue of HybridFCL.  Provides a variational sampler and a QNN
    that together emulate the classical encoder, sampler head, and classifier.
    The ``run`` method accepts a list of weight parameters and returns
    (qnn_outputs, sampler_probs) as NumPy arrays.
    """

    def __init__(self,
                 num_qubits: int = 1,
                 depth: int = 1,
                 backend=None,
                 shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Build the underlying circuit and metadata
        self.circuit, self.encoding_params, self.weight_params, self.observables = \
            build_classifier_circuit(num_qubits, depth)

        # Sampler and QNN primitives
        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(circuit=self.circuit,
                                      input_params=self.encoding_params,
                                      weight_params=self.weight_params,
                                      sampler=self.sampler)
        self.qnn = QNN(circuit=self.circuit,
                       input_params=self.encoding_params,
                       weight_params=self.weight_params,
                       sampler=self.sampler)

    def run(self, thetas: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute the sampler and QNN with the supplied weight parameters.
        Returns:
            qnn_outputs: expectation values from the QNN (shape (num_qubits,))
            sampler_probs: softmax‑like probabilities from the sampler (shape (2,))
        """
        # Convert to list for indexing
        theta_list = list(thetas)
        # Sampler returns a probability distribution over 2 outcomes
        probs = self.sampler_qnn(theta_list)
        # QNN returns expectation values for each observable
        outputs = self.qnn(theta_list)
        return np.array(outputs), np.array(probs)

__all__ = ["HybridFCL"]
