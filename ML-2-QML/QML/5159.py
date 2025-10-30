"""Quantum module implementing a hybrid quanvolution filter using Qiskit."""
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int):
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Input encoding
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

class QuanvolutionHybridModel:
    """Quantum kernel that processes 2×2 patches via a variational circuit."""
    def __init__(self, num_qubits: int = 4, depth: int = 2, shots: int = 1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.circuit, self.encoding_params, self.weight_params, self.observables = \
            build_classifier_circuit(num_qubits, depth)
        self.backend = Aer.get_backend("aer_simulator")

    def _expectations(self, bound_circuit: QuantumCircuit) -> np.ndarray:
        """Return expectation values of Z on each qubit."""
        state = Statevector.from_instruction(bound_circuit)
        return np.array([state.expectation_value(obs).real for obs in self.observables])

    def forward(self, patches: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each patch.

        Parameters
        ----------
        patches : np.ndarray
            Shape (batch, num_patches, num_qubits) with pixel values in [0, 1].

        Returns
        -------
        np.ndarray
            Shape (batch, num_patches, num_qubits) of expectation values.
        """
        batch, num_patches, _ = patches.shape
        outputs = np.empty((batch, num_patches, self.num_qubits), dtype=np.float32)

        for b in range(batch):
            for p in range(num_patches):
                binding = {param: val for param, val in zip(self.encoding_params, patches[b, p])}
                bound_circuit = self.circuit.bind_parameters(binding)
                outputs[b, p] = self._expectations(bound_circuit)
        return outputs
