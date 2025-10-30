from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridEstimator(EstimatorQNN):
    """
    Quantum hybrid estimator that mirrors the classical quanvolution pipeline
    using a 2‑qubit variational circuit.  One qubit encodes the input value,
    the other encodes the trainable weight.  The circuit is followed by an
    entangling layer and a Y‑observable to produce an expectation value.
    """
    def __init__(self, input_dim: int = 1, weight_dim: int = 1,
                 observable: SparsePauliOp | None = None) -> None:
        # Create parameters for inputs and weights
        input_params = [Parameter(f"inp_{i}") for i in range(input_dim)]
        weight_params = [Parameter(f"w_{j}") for j in range(weight_dim)]

        # Define a minimal 2‑qubit variational circuit
        qc = QuantumCircuit(input_dim + weight_dim)
        # Encode inputs with Ry rotations
        for i, p in enumerate(input_params):
            qc.ry(p, i)
        # Encode weights with Rx rotations
        for j, p in enumerate(weight_params):
            qc.rx(p, input_dim + j)
        # Entangling layer
        qc.cx(0, 1)
        # Additional variational rotations
        qc.rz(weight_params[0], 0)
        qc.rz(weight_params[0], 1)

        # Default observable if none provided
        if observable is None:
            observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1.0)])

        # Use a state‑vector estimator for simulation
        estimator = StatevectorEstimator()

        super().__init__(circuit=qc,
                         observables=observable,
                         input_params=input_params,
                         weight_params=weight_params,
                         estimator=estimator)

__all__ = ["HybridEstimator"]
