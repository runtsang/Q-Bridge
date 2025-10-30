import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli

class HybridNATModel:
    """Quantum hybrid model with amplitude encoding, variational layers and linear post‑processing."""
    def __init__(self, n_qubits=4, n_layers=2, n_classes=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.qreg = QuantumRegister(n_qubits)
        self.creg = ClassicalRegister(n_qubits)
        self.circuit = QuantumCircuit(self.qreg, self.creg)
        self.params = []
        self._build_circuit()

        # Simple linear head: weights and bias
        self.head_weights = np.random.randn(n_classes, n_qubits) * 0.1
        self.head_bias = np.zeros(n_classes)

    def _build_circuit(self):
        # Parameterized encoding placeholder
        for i in range(self.n_qubits):
            theta = Parameter(f'theta_{i}')
            self.circuit.ry(theta, self.qreg[i])
            self.params.append(theta)
        # Variational layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                phi = Parameter(f'phi_{i}_{layer}')
                self.circuit.rz(phi, self.qreg[i])
                self.params.append(phi)
            for i in range(self.n_qubits - 1):
                self.circuit.cx(self.qreg[i], self.qreg[i + 1])

    def bind_params(self, param_values):
        if len(param_values)!= len(self.params):
            raise ValueError('Parameter count mismatch')
        mapping = dict(zip(self.params, param_values))
        return self.circuit.bind_parameters(mapping)

    def _measure_expectations(self, bound_circuit, observables):
        state = Statevector.from_instruction(bound_circuit)
        return [state.expectation_value(obs) for obs in observables]

    def evaluate(self, observables, parameter_sets):
        """Compute expectation values for each parameter set."""
        results = []
        for params in parameter_sets:
            bound = self.bind_params(params)
            row = self._measure_expectations(bound, observables)
            results.append(row)
        return results

    def predict(self, input_features, param_values):
        """Encode input features via Ry angles, evaluate, and apply linear head."""
        # Encode features: simple angle encoding
        encoded_angles = np.arcsin(np.clip(input_features, -1, 1))
        # Build full parameter list: encoding + variational params
        full_params = list(encoded_angles) + list(param_values)
        bound = self.bind_params(full_params)
        state = Statevector.from_instruction(bound)
        # Expectation values of Z on each qubit
        expectations = []
        for i in range(self.n_qubits):
            pauli_str = 'I' * i + 'Z' + 'I' * (self.n_qubits - i - 1)
            expectations.append(state.expectation_value(Pauli(pauli_str)))
        # Linear head
        logits = self.head_weights @ expectations + self.head_bias
        return logits
