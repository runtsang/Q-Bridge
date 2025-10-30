import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QuantumEstimator

class HybridEstimatorQNN(nn.Module):
    """
    Classical‑quantum hybrid regressor.

    Architecture:
    * Convolutional feature extractor (1→4 channels, 2×2 kernel, stride 2)
    * Linear head to map features to *num_qubits* values
    * Variational quantum circuit that takes these values as input parameters
      and returns the expectation of a Y observable.
    """

    def __init__(self, input_channels: int = 1, num_qubits: int = 1, output_dim: int = 1):
        super().__init__()
        # Classical feature extractor (mirrors quanvolution filter)
        self.conv = nn.Conv2d(input_channels, 4, kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4 * 14 * 14, num_qubits)

        # Quantum part: parameters and circuit
        self.num_qubits = num_qubits
        self.input_params = [Parameter(f"x_{i}") for i in range(num_qubits)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(num_qubits)]
        self.circuit = self._build_circuit()
        self.observables = SparsePauliOp.from_list([("Y" * num_qubits, 1.0)])
        self.estimator = QuantumEstimator()

    def _build_circuit(self) -> QuantumCircuit:
        """Build a simple variational circuit with input encoding and weight gates."""
        qc = QuantumCircuit(self.num_qubits)
        # Input encoding (Ry)
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        # Weight encoding (Rx) followed by a layer of entanglement
        for i, p in enumerate(self.weight_params):
            qc.rx(p, i)
        if self.num_qubits > 1:
            qc.cx(0, 1)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Quantum expectation values of shape (batch, output_dim).
        """
        # Classical feature extraction
        features = self.conv(x)          # (batch, 4, 14, 14)
        flat = self.flatten(features)    # (batch, 4*14*14)
        # Map to quantum input values
        q_inputs = self.linear(flat)     # (batch, num_qubits)

        # Evaluate quantum circuit for each batch element
        expectations = []
        for q_in in q_inputs:
            # Bind classical inputs to quantum circuit
            bind_dict = {p: float(v) for p, v in zip(self.input_params, q_in.tolist())}
            bound_qc = self.circuit.bind_parameters(bind_dict)
            # Estimate expectation value
            result = self.estimator.run(
                circuits=[bound_qc],
                observables=[self.observables]
            ).result()
            exp_val = result.quasi_results[0].expectation
            expectations.append(exp_val)
        return torch.tensor(expectations, device=x.device).unsqueeze(-1)

    def get_params(self):
        """Return all trainable parameters (classical + quantum weights)."""
        return list(self.parameters()) + self.weight_params

__all__ = ["HybridEstimatorQNN"]
