import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

class FraudDetectionQuantumKernel:
    """
    Quantum kernel that maps 4‑dimensional classical features to
    expectation values of a simple 2‑qubit circuit.
    """
    def __init__(self):
        self.params = [Parameter(f"theta_{i}") for i in range(4)]
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.params[0], 0)
        self.circuit.ry(self.params[1], 1)
        self.circuit.rx(self.params[2], 0)
        self.circuit.rx(self.params[3], 1)

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum kernel for a batch of 4D inputs.
        Returns a numpy array of shape (batch, 4) containing
        expectation values for each input.
        """
        batch = inputs.shape[0]
        results = np.zeros((batch, 4))
        for i in range(batch):
            param_dict = dict(zip(self.params, inputs[i]))
            qc = self.circuit.assign_parameters(param_dict)
            state = Statevector.from_instruction(qc)
            # expectation values of Pauli Z on each qubit
            results[i, 0] = state.expectation_value(Statevector.from_label("Z0"))
            results[i, 1] = state.expectation_value(Statevector.from_label("Z1"))
            # two‑qubit correlation Z⊗Z
            results[i, 2] = state.expectation_value(Statevector.from_label("Z0Z1"))
            results[i, 3] = 1.0  # identity
        return results

class FastEstimator:
    """
    Lightweight estimator that evaluates expectation values of a
    parametrized circuit using Qiskit’s Statevector simulator.
    """
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def evaluate(self, observables, parameter_sets):
        results = []
        for values in parameter_sets:
            bound = self.circuit.assign_parameters(dict(zip(self.params, values)))
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["FraudDetectionQuantumKernel", "FastEstimator"]
