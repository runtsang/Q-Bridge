"""Quantum neural network with two qubits and variational layers."""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator

class SharedClassName:
    """Variational quantum circuit for regression with 2 qubits.

    Parameters
    ----------
    n_layers : int, optional
        Number of variational layers. Defaults to 2.
    """
    def __init__(self, n_layers: int = 2):
        self.n_layers = n_layers
        # Input parameters (one per qubit)
        self.input_params = [Parameter(f"input{i}") for i in range(2)]
        # Weight parameters: one per qubit per layer
        self.weight_params = [Parameter(f"w{l}_{q}") for l in range(n_layers) for q in range(2)]
        self.circuit = self._build_circuit()
        self.observables = SparsePauliOp.from_list([("Z" * self.circuit.num_qubits, 1)])
        self.estimator = Estimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a layered entangling circuit."""
        qc = QuantumCircuit(2)
        for l in range(self.n_layers):
            # Encode inputs
            qc.ry(self.input_params[0], 0)
            qc.ry(self.input_params[1], 1)
            # Variational rotation layer
            qc.rz(self.weight_params[l*2], 0)
            qc.rz(self.weight_params[l*2 + 1], 1)
            # Entanglement
            qc.cx(0, 1)
            qc.cx(1, 0)
        return qc

    def expectation(self,
                    input_vals: list[float],
                    weight_vals: list[float]) -> float:
        """Compute expectation value for given inputs and weights."""
        param_dict = {p: v for p, v in zip(self.input_params + self.weight_params,
                                          input_vals + weight_vals)}
        result = self.estimator_qnn.run(inputs=[param_dict])
        return float(result[0])

def EstimatorQNN() -> SharedClassName:
    """Compatibility wrapper returning the default SharedClassName."""
    return SharedClassName()

__all__ = ["EstimatorQNN", "SharedClassName"]
