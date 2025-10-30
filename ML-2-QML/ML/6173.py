"""Hybrid classical‑quantum classifier with a learnable feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

# Import the quantum circuit builder from the QML module
from.quantum_factory import QuantumClassifierModel as QMLQuantumClassifierModel

__all__ = ["QuantumClassifierModel"]

class QuantumClassifierModel(nn.Module):
    """
    PyTorch module that combines a classical feature extractor with a variational quantum circuit.
    The quantum part is evaluated via a custom autograd function that wraps Qiskit simulation.
    """

    def __init__(self, num_features: int, num_qubits: int, quantum_depth: int, hidden_dim: int = 32):
        super().__init__()
        # Classical feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits),
        )
        self.quantum_depth = quantum_depth
        self.num_qubits = num_qubits

        # Instantiate the quantum circuit builder
        qml_builder = QMLQuantumClassifierModel(num_qubits, quantum_depth)
        self.circuit, self.encoding_params, self.weight_params, self.observables = qml_builder.get_circuit()

        # Learnable weights for the variational circuit
        self.weight_vector = nn.Parameter(torch.randn(len(self.weight_params)))
        # Linear head to produce logits
        self.output_head = nn.Linear(num_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features -> quantum expectation values -> logits.
        This implementation assumes a single sample per forward call.
        """
        features = self.feature_extractor(x)
        expectations = QuantumCircuitForward.apply(
            features,
            self.weight_vector,
            self.circuit,
            self.encoding_params,
            self.weight_params,
            self.observables,
        )
        logits = self.output_head(expectations)
        return logits

class QuantumCircuitForward(Function):
    """
    Autograd Function that evaluates a Qiskit circuit and returns expectation values.
    Gradients are approximated using a finite‑difference method for simplicity.
    """

    @staticmethod
    def forward(ctx, data: torch.Tensor, weights: torch.Tensor,
                circuit, encoding_params, weight_params, observables):
        import numpy as np
        from qiskit import Aer, transpile

        backend = Aer.get_backend('statevector_simulator')

        # Bind parameters
        param_dict = {str(p): float(v) for p, v in zip(encoding_params, data.tolist())}
        param_dict.update({str(p): float(v) for p, v in zip(weight_params, weights.tolist())})
        bound_circuit = circuit.bind_parameters(param_dict)

        # Simulate
        job = backend.run(transpile(bound_circuit, backend))
        result = job.result()
        statevector = result.get_statevector(bound_circuit)

        # Compute expectation values
        expectations = []
        for obs in observables:
            mat = obs.to_matrix()
            exp = np.real(np.vdot(statevector, mat @ statevector))
            expectations.append(exp)

        expectations = torch.tensor(expectations, dtype=data.dtype, device=data.device)
        ctx.save_for_backward(data, weights)
        ctx.circuit = circuit
        ctx.encoding_params = encoding_params
        ctx.weight_params = weight_params
        ctx.observables = observables
        ctx.backend = backend
        return expectations

    @staticmethod
    def backward(ctx, grad_output):
        data, weights = ctx.saved_tensors
        eps = 1e-3
        grad_data = torch.zeros_like(data)
        grad_weights = torch.zeros_like(weights)

        # Gradient w.r.t. data (input features)
        for i in range(data.shape[0]):
            data_shift = data.clone()
            data_shift[i] += eps
            out_plus = QuantumCircuitForward.apply(
                data_shift, weights, ctx.circuit, ctx.encoding_params, ctx.weight_params, ctx.observables
            )
            data_shift[i] -= 2 * eps
            out_minus = QuantumCircuitForward.apply(
                data_shift, weights, ctx.circuit, ctx.encoding_params, ctx.weight_params, ctx.observables
            )
            grad_data[i] = ((out_plus - out_minus) / (2 * eps)).dot(grad_output)

        # Gradient w.r.t. variational weights
        for i in range(weights.shape[0]):
            weights_shift = weights.clone()
            weights_shift[i] += eps
            out_plus = QuantumCircuitForward.apply(
                data, weights_shift, ctx.circuit, ctx.encoding_params, ctx.weight_params, ctx.observables
            )
            weights_shift[i] -= 2 * eps
            out_minus = QuantumCircuitForward.apply(
                data, weights_shift, ctx.circuit, ctx.encoding_params, ctx.weight_params, ctx.observables
            )
            grad_weights[i] = ((out_plus - out_minus) / (2 * eps)).dot(grad_output)

        return grad_data, grad_weights, None, None, None, None
