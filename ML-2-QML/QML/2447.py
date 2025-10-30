import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator as QiskitEstimator

class SharedClassName:
    """Quantum neural network that embeds a self‑attention style circuit
    followed by a linear layer.  The attention block is a parameterised
    rotation plus controlled‑X entanglement pattern that mirrors the
    classical attention mechanism.  The final expectation value is
    evaluated with a StatevectorEstimator, enabling gradient‑based
    optimisation."""
    def __init__(self, n_qubits: int = 4, input_dim: int = 1):
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self._build_circuit()

    def _build_circuit(self):
        # Parameters for rotation and entanglement in the attention block
        self.rotation_params = [Parameter(f"rot_{i}") for i in range(3 * self.n_qubits)]
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(self.n_qubits - 1)]
        # Build attention circuit
        self.attention_circuit = self._build_attention_circuit()
        # Build estimator circuit (simple linear layer)
        self.weight_params = [Parameter(f"w_{i}") for i in range(self.n_qubits)]
        self.estimator_circuit = self._build_estimator_circuit()
        # Combine into a single circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.compose(self.attention_circuit, inplace=True)
        self.circuit.compose(self.estimator_circuit, inplace=True)
        # Observable for regression output
        self.observable = SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])
        # Estimator primitive
        self.estimator = QiskitEstimator(backend=AerSimulator(method="statevector"))

    def _build_attention_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        return qc

    def _build_estimator_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i, w in enumerate(self.weight_params):
            qc.ry(w, i)
        return qc

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Evaluate the quantum circuit for a batch of inputs."""
        results = []
        for x in inputs:
            param_bindings = {}
            # Bind rotation parameters to input features
            for idx, param in enumerate(self.rotation_params):
                # Map each input dimension to a rotation; if more rotations than inputs, pad with zeros
                dim = idx // 3
                param_bindings[param] = float(x[dim]) if dim < self.input_dim else 0.0
            # Bind entanglement parameters to zero (or a small constant)
            for param in self.entangle_params:
                param_bindings[param] = 0.0
            # Bind weight parameters to zero (initially)
            for param in self.weight_params:
                param_bindings[param] = 0.0
            bound_circ = self.circuit.bind_parameters(param_bindings)
            result = self.estimator.run(bound_circ, [self.observable])
            exp_val = result.values[0]
            results.append(exp_val)
        return np.array(results)

__all__ = ["SharedClassName"]
