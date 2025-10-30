"""
HybridEstimatorQNN: Variational quantum circuit for regression tasks.
The circuit uses multiple qubits, parameterized rotation layers, and
entanglement via CX gates. It returns expectation values of a set
of observables that are trained to approximate a regression target.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator

class HybridEstimatorQNN:
    """
    Construct a variational quantum circuit with:
    - 3 qubits
    - 2 layers of rotation gates
    - Entanglement via CX gates
    - Observable: sum of Z on each qubit
    """

    def __init__(self, num_qubits: int = 3, num_layers: int = 2,
                 estimator: Estimator = None) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.estimator = estimator or Estimator()
        self.circuit = self._build_circuit()
        self.input_params, self.weight_params = self._extract_params()
        self.observables = self._build_observables()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Input layer
        for i in range(self.num_qubits):
            param = Parameter(f"x_{i}")
            qc.ry(param, i)
        # Variational layers
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                param_t = Parameter(f"theta_{layer}_{i}")
                qc.ry(param_t, i)
                param_p = Parameter(f"phi_{layer}_{i}")
                qc.rz(param_p, i)
            # Entangling CX
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(self.num_qubits - 1, 0)
        return qc

    def _extract_params(self):
        input_params = [p for p in self.circuit.parameters if p.name.startswith("x_")]
        weight_params = [p for p in self.circuit.parameters
                         if p.name.startswith("theta_") or p.name.startswith("phi_")]
        return input_params, weight_params

    def _build_observables(self):
        obs = []
        for i in range(self.num_qubits):
            pauli_str = "Z" * self.num_qubits
            pauli_str = pauli_str[:i] + "Z" + pauli_str[i+1:]
            obs.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
        return obs

    def get_expectations(self, inputs, weights):
        """
        Evaluate the circuit with given inputs and weights.
        Returns the expectation values of the observables.
        """
        param_dict = dict(zip(self.input_params + self.weight_params,
                              inputs + weights))
        return self.estimator_qnn.run(param_dict)

def EstimatorQNN() -> HybridEstimatorQNN:
    """Factory returning an instance of the enhanced quantum estimator."""
    return HybridEstimatorQNN()

__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
