from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNN:
    """A variational quantum circuit that extends the original single‑qubit design.

    Enhancements:
    - Two‑qubit circuit with CNOT entanglement
    - Parameterised RX (for inputs) and RZ (for weights) layers per qubit
    - Two observables (Y⊗I and I⊗Z) to give richer feature space
    - Uses a state‑vector estimator for exact evaluation
    """
    def __init__(self, input_dim: int = 2, weight_dim: int = 2):
        # Define parameters
        self.input_params = [Parameter(f"input_{i}") for i in range(input_dim)]
        self.weight_params = [Parameter(f"weight_{i}") for i in range(weight_dim)]
        # Build circuit
        self.circuit = QuantumCircuit(input_dim)
        # Encode inputs
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i)
        # Entanglement
        self.circuit.cx(0, 1)
        # Trainable rotations (weights)
        for i, p in enumerate(self.weight_params):
            self.circuit.rz(p, i)
        # Build observables
        obs_list = [("Y" * input_dim, 1), ("Z" * input_dim, 1)]
        self.observables = SparsePauliOp.from_list(obs_list)
        # Instantiate estimator
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def __call__(self, inputs: list[float] | tuple[float, float]) -> float:
        """Convenience wrapper that accepts a 2‑element input tuple and returns the prediction."""
        return self.estimator_qnn(inputs)

__all__ = ["EstimatorQNN"]
