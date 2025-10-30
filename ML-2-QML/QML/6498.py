import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from typing import Iterable

class HybridEstimatorQNN:
    """
    Quantum implementation of the hybrid estimator.  Builds a 1‑qubit
    circuit with an input rotation Ry(θ) and a trainable rotation Rx(ϕ).
    The Y observable expectation is used as the feature for a classical
    regression head.  The circuit is wrapped in Qiskit's EstimatorQNN so
    that it can be trained with gradient‑based optimisers.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8) -> None:
        # Define parameters
        self.input_params = [Parameter(f"θ_{i}") for i in range(input_dim)]
        self.weight_params = [Parameter(f"ϕ_{i}") for i in range(hidden_dim)]

        # Build the quantum circuit
        self.circuit = QuantumCircuit(input_dim)
        for q in range(input_dim):
            self.circuit.h(q)
            self.circuit.ry(self.input_params[q], q)
            self.circuit.rx(self.weight_params[q], q)

        # Observable: Pauli‑Y on the first qubit
        self.observable = SparsePauliOp.from_list([("Y" * input_dim, 1)])

        # Estimator
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit on a simulator and return the expectation
        value of the Y observable as a NumPy array.  The `thetas` list
        should contain values for both input and weight parameters.
        """
        param_bind = {p: t for p, t in zip(self.input_params + self.weight_params, thetas)}
        job = execute(
            self.circuit,
            Aer.get_backend("qasm_simulator"),
            shots=100,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()), dtype=float)
        probs = counts / 100
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridEstimatorQNN"]
