import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumSelfAttentionSamplerClassifier:
    """Quantum version of the hybrid attention classifier.
    Implements a self‑attention style circuit followed by a
    parameterised SamplerQNN for binary classification."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameter vectors
        self.inputs = ParameterVector("input", n_qubits)
        self.weights = ParameterVector("weight", n_qubits * 3)

        # Build circuit
        self.circuit = self._build_circuit()

        # Sampler QNN
        self.sampler = StatevectorSampler(self.backend)
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=self.inputs,
                              weight_params=self.weights,
                              sampler=self.sampler)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Input rotations
        for i in range(self.n_qubits):
            qc.ry(self.inputs[i], i)
        # Entanglement layer (CRX style)
        for i in range(self.n_qubits - 1):
            qc.crx(self.weights[3 * i], i, i + 1)
        # Additional parameterised rotations
        for i in range(self.n_qubits):
            qc.rx(self.weights[3 * i], i)
            qc.ry(self.weights[3 * i + 1], i)
            qc.rz(self.weights[3 * i + 2], i)
        qc.measure_all()
        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Bind input parameters
        param_bindings = {p: v for p, v in zip(self.inputs, inputs[:self.n_qubits])}
        # Randomly initialise weight parameters (could be learned)
        param_bindings.update({p: np.random.uniform(-np.pi, np.pi)
                               for p in self.weights})
        # Execute SamplerQNN
        result = self.qnn(param_bindings)
        # Convert counts to probability of the first qubit being |1>
        probs = np.array([int(k[0]) for k in result.keys()])
        probs = probs / self.shots
        # Aggregate into a binary probability
        prob_pos = probs.mean()
        return np.array([prob_pos, 1 - prob_pos])

__all__ = ["QuantumSelfAttentionSamplerClassifier"]
