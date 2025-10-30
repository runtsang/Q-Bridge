import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

class HybridEstimatorQNN:
    """
    Quantum self‑attention block that produces expectation values which
    can be used as features for a downstream classical model.  The circuit
    consists of parameterized single‑qubit rotations followed by
    controlled‑R‑X entangling gates, mirroring the structure of the
    classical attention module.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits

        # Parameterised rotations (3 per qubit)
        self.rotation_params = [Parameter(f"rot_{i}") for i in range(n_qubits * 3)]
        # Entanglement parameters (one per adjacent pair)
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(n_qubits - 1)]

        # Build the circuit once; parameters will be bound during prediction
        self.circuit = self._build_circuit()

        # Observable for expectation value extraction
        self.observables = SparsePauliOp.from_list([(("Y" * n_qubits), 1)])

        # Instantiate the Qiskit Machine Learning EstimatorQNN
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.rotation_params,
            weight_params=self.entangle_params,
            estimator=self.estimator
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        return qc

    def run(
        self,
        input_values: np.ndarray,
        weight_values: np.ndarray,
        shots: int = 1024
    ) -> np.ndarray:
        """
        Execute the quantum circuit with given parameter values and return
        the expectation value of the observable.

        Args:
            input_values: Array of length n_qubits*3 containing rotation parameters.
            weight_values: Array of length n_qubits-1 containing entanglement parameters.
            shots: Number of measurement shots for the simulator.

        Returns:
            Expectation value array of shape (1,).
        """
        param_dict = {
            param: value
            for param, value in zip(self.rotation_params, input_values)
        }
        param_dict.update(
            {param: value for param, value in zip(self.entangle_params, weight_values)}
        )
        result = self.estimator_qnn.predict(param_dict, shots=shots)
        return result

__all__ = ["HybridEstimatorQNN"]
