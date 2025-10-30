import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class HybridQuantumHead(nn.Module):
    """
    Hybrid head that forwards activations through a quantum circuit.
    The circuit is a simple variational circuit that takes the input features as
    parameters for Ry gates and measures the expectation value of Z on the first qubit.
    """
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.shift = shift
        self.backend = AerSimulator()
        self.shots = 100
        # Build a simple parameterised circuit
        self.circuit = QuantumCircuit(in_features)
        self.params = [Parameter(f"theta_{i}") for i in range(in_features)]
        for i in range(in_features):
            self.circuit.ry(self.params[i], i)
        self.circuit.measure_all()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that evaluates the quantum circuit for each input in the batch.
        """
        # Convert inputs to a numpy array
        thetas = inputs.detach().cpu().numpy()
        expectations = []
        for theta in thetas:
            # Bind parameters to the circuit
            param_dict = {self.params[i]: theta[i] for i in range(self.in_features)}
            bound_circuit = self.circuit.bind_parameters(param_dict)
            compiled = transpile(bound_circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            # Compute expectation of Z on the first qubit
            exp = 0.0
            for bitstring, count in counts.items():
                z = 1 if bitstring[0] == '0' else -1
                exp += z * count
            exp /= self.shots
            expectations.append(exp)
        # Convert to torch tensor
        return torch.tensor(expectations, dtype=torch.float32, device=inputs.device)

__all__ = ["HybridQuantumHead"]
