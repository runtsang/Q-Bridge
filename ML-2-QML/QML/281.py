"""Quantum variational estimator for regression.

The circuit uses 3 qubits with a layered Ansatz:
    Ry(input) -> Rz(weight) -> CNOT entanglement.
Observables are the Pauli Y of each qubit; expectation values
are summed to produce a scalar output.  The estimator supports
state‑vector simulation and automatic gradient via the
parameter‑shift rule.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNN:
    def __init__(self, shots: int = 1024, backend=None) -> None:
        self._shots = shots
        self._backend = backend

        # Define parameters
        self.input_params = [Parameter(f"input_{i}") for i in range(3)]
        self.weight_params = [Parameter(f"weight_{i}") for i in range(3)]

        # Build variational circuit
        qc = QuantumCircuit(3)
        for i in range(3):
            qc.ry(self.input_params[i], i)
            qc.rz(self.weight_params[i], i)
        # Entangling layer
        qc.cx(0, 1)
        qc.cx(1, 2)

        # Observables: Pauli Y on each qubit
        self.observable = SparsePauliOp.from_list(
            [("Y0", 1), ("Y1", 1), ("Y2", 1)]
        )

        # Estimator primitive
        self.estimator = StatevectorEstimator(backend=self._backend, shots=self._shots)

        # Wrap into Qiskit EstimatorQNN
        self.qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def __call__(self, inputs: list[float], weights: list[float]) -> float:
        """Evaluate the quantum neural network."""
        param_dict = dict(zip(self.input_params + self.weight_params,
                              inputs + weights))
        result = self.qnn.predict(param_dict)
        return float(result[0])

    def gradient(self, inputs: list[float], weights: list[float]) -> list[float]:
        """Compute the gradient w.r.t. weight parameters using
        the parameter‑shift rule implemented by Qiskit.
        """
        param_dict = dict(zip(self.input_params + self.weight_params,
                              inputs + weights))
        grads = self.qnn.gradient(param_dict)
        # grads[0] corresponds to input gradients; skip them
        return [float(g) for g in grads[1:4]]

__all__ = ["EstimatorQNN"]
