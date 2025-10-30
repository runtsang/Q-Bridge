import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class HybridFCL:
    """A parameterised quantum circuit that emulates a fully‑connected layer.

    The design combines the single‑qubit circuit of the original FCL with
    the multi‑parameter EstimatorQNN pattern.  Each qubit receives an H gate,
    followed by a Ry(θ) and Rx(φ) rotation whose angles are treated as
    trainable weights.  The observable is a tensor‑product of Pauli‑Y
    operators, and the expectation value is returned as a NumPy array.
    The class exposes a ``run`` method that accepts a list of parameter
    values and evaluates the circuit on the chosen backend.
    """
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Define symbolic parameters for each qubit
        self.input_params = [Parameter(f"θ{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"φ{i}") for i in range(n_qubits)]

        # Build the circuit
        self._circuit = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self._circuit.h(q)
            self._circuit.ry(self.input_params[q], q)
            self._circuit.rx(self.weight_params[q], q)
        self._circuit.measure_all()

        # Observable: tensor product of Pauli‑Y on all qubits
        pauli_list = ["Y" * n_qubits]
        self.observable = SparsePauliOp.from_list(pauli_list)

        # Wrap with EstimatorQNN
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self._circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(self, params: list[float]) -> np.ndarray:
        """Evaluate the quantum circuit with the supplied parameter list.

        Parameters are expected in the order [θ0, φ0, θ1, φ1,...].
        """
        if len(params)!= 2 * self.n_qubits:
            raise ValueError(f"Expected {2*self.n_qubits} parameters, got {len(params)}")
        param_bindings = {p: v for p, v in zip(self.input_params + self.weight_params, params)}
        result = self.estimator_qnn.forward(param_bindings)
        return np.array([result])

__all__ = ["HybridFCL"]
