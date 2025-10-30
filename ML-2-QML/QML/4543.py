"""HybridEstimatorQNN: a quantum estimator that mirrors the classical architecture.
The circuit encodes input features into rotation gates, applies a variational
entanglement layer (CRX) and measures a Pauli‑Y observable.  A StatevectorEstimator
returns the expectation value, which is used as the regression output.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
import numpy as np

class HybridEstimatorQNN:
    """Parameterized quantum circuit with input, weight and entanglement parameters."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # Parameter placeholders
        self.input_params  = [Parameter(f"inp_{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"wt_{i}")  for i in range(n_qubits)]
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(n_qubits-1)]

        # Build the variational layers
        for i in range(n_qubits):
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        for i in range(n_qubits-1):
            self.circuit.crx(self.entangle_params[i], i, i+1)

        self.circuit.measure_all()

        # Backend and estimator
        self.backend = Aer.get_backend("statevector_simulator")
        self.estimator = StatevectorEstimator(self.backend)

    def run(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return the expectation value of Pauli‑Y for the given input and weight vectors."""
        # Bind parameters
        bind_dict = {p: v for p, v in zip(self.input_params, inputs)}
        bind_dict.update({p: w for p, w in zip(self.weight_params, weights)})

        # Execute
        result = self.estimator(self.circuit, parameter_binds=[bind_dict])
        # Expectation of Y on all qubits, summed
        expectation = sum([abs(r) for r in result.expectation_values])
        return np.array([expectation])

__all__ = ["HybridEstimatorQNN"]
