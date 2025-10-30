import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

class HybridEstimatorQNN:
    """Quantum‑classical estimator that uses a self‑attention style circuit
    to encode the input features into expectation values, which are then
    read out by a classical linear layer."""
    def __init__(self, input_dim: int = 2, n_qubits: int = 4):
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        # Parameters for rotations and entanglement
        self.rotation_params = [Parameter(f"rot_{i}") for i in range(3 * n_qubits)]
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(n_qubits - 1)]
        # Build the circuit template
        self.circuit = self._build_circuit()
        # Observable: Y on all qubits
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])
        # Backend for simulation
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)
        # Encode inputs as rotations on first input_dim qubits
        for i in range(self.input_dim):
            qc.ry(self.rotation_params[i], i)
        # Remaining rotation gates (learnable)
        for i in range(self.input_dim, 3 * self.n_qubits):
            qc.rx(self.rotation_params[i], i % self.n_qubits)
        # Entanglement layer (CRX)
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        qc.measure(qr, cr)
        return qc

    def run(self, input_values: np.ndarray, shots: int = 1024) -> float:
        """Execute the quantum circuit on the simulator and return the
        expectation value of the observable."""
        # Bind input values to the first rotation parameters
        bound_params = {self.rotation_params[i]: input_values[i] for i in range(self.input_dim)}
        bound_circuit = self.circuit.bind_parameters(bound_params)
        job = execute(bound_circuit, self.backend, shots=shots)
        counts = job.result().get_counts(bound_circuit)
        # Convert counts to expectation value of Y observable (simplified)
        exp = 0.0
        for outcome, freq in counts.items():
            parity = (-1) ** (outcome.count('1'))
            exp += parity * freq
        exp /= shots
        return exp

def EstimatorQNN() -> HybridEstimatorQNN:
    """Factory function that returns a hybrid estimator identical to the original API."""
    return HybridEstimatorQNN()

__all__ = ["EstimatorQNN", "HybridEstimatorQNN"]
