from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import numpy as np

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, param_values):
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables, parameter_sets):
        observables = list(observables)
        results = []
        for params in parameter_sets:
            bound_circ = self._bind(params)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot‑noise to deterministic expectation values."""
    def evaluate(self, observables, parameter_sets, *, shots=None, seed=None):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(complex(val).real, max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

class EnhancedSamplerQNN:
    """
    Parameterised quantum sampler with a fast estimator interface.
    """
    def __init__(self):
        # Parameter vectors
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 6)

        # Build circuit
        qc = QuantumCircuit(2)
        # Input encoding
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        # Entanglement and rotation layers
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(1, 0)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[4], 0)
        qc.ry(self.weights[5], 1)

        self.circuit = qc
        self.estimator = FastEstimator(self.circuit)

    def evaluate(self, observables, parameter_sets, *, shots=None, seed=None):
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["EnhancedSamplerQNN"]
