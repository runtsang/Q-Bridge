import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNN:
    """
    Variational quantum circuit that supports multiple qubits, a
    configurable number of layers, and a set of input and trainable
    parameters.  The circuit is wrapped in a qiskit_machine_learning
    EstimatorQNN to provide expectation values of a Pauli‑Z
    observable.  It is a quantum analogue of the classical
    feed‑forward regressor.
    """

    def __init__(
        self,
        num_qubits: int = 1,
        num_layers: int = 2,
        input_dim: int = 1,
        weight_decay: float = 0.0,
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.weight_decay = weight_decay

        # Input parameters (one per input dimension)
        self.input_params = [Parameter(f"x{i}") for i in range(input_dim)]

        # Weight parameters: 3 per qubit per layer (ry, rz, entanglement)
        self.weight_params = [
            Parameter(f"theta_{l}_{q}_{t}")
            for l in range(num_layers)
            for q in range(num_qubits)
            for t in ("ry", "rz", "ent")
        ]

        # Build circuit
        self.circuit = QuantumCircuit(num_qubits)

        # Encode inputs (angle encoding)
        for q in range(num_qubits):
            self.circuit.ry(self.input_params[0], q)

        idx = 0
        for l in range(num_layers):
            # Local rotations
            for q in range(num_qubits):
                self.circuit.ry(self.weight_params[idx], q)
                idx += 1
            for q in range(num_qubits):
                self.circuit.rz(self.weight_params[idx], q)
                idx += 1
            # Entanglement layer
            for q in range(num_qubits):
                self.circuit.rz(self.weight_params[idx], q)
                idx += 1
            for q in range(num_qubits - 1):
                self.circuit.cx(q, q + 1)
            self.circuit.cx(num_qubits - 1, 0)

        # Observable: sum of Z on all qubits
        pauli_z = Pauli("Z" * num_qubits)
        self.observable = SparsePauliOp.from_list([(pauli_z, 1.0)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the variational circuit for the given inputs.
        Supports single samples or batched input arrays.
        """
        if inputs.ndim == 1:
            params = {p: float(v) for p, v in zip(self.input_params, inputs)}
            return self.estimator_qnn(params)
        else:
            results = []
            for sample in inputs:
                params = {p: float(v) for p, v in zip(self.input_params, sample)}
                results.append(self.estimator_qnn(params))
            return np.array(results)

    def l2_regularization(self) -> float:
        """Placeholder for weight‑decay regularisation."""
        return 0.0

__all__ = ["EstimatorQNN"]
