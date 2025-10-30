"""
FraudDetectionHybridQuantum.py

Quantum counterpart of FraudDetectionHybrid that replaces the linear classifier
with a variational quantum circuit (VQC) executed on Qiskit.  The VQC receives
the classical embedding produced by the photonic‑style layers and the
self‑attention block, then outputs a single expectation value that is
converted to a probability.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter


# --------------------------------------------------------------------------- #
# Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumExpectationCircuit:
    """
    Parameterised 4‑qubit circuit that computes an expectation value of Z⊗I⊗I⊗I.
    The parameters are fed as rotation angles for each qubit.
    """

    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = [Parameter(f"θ_{i}") for i in range(n_qubits)]
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        for i, θ in enumerate(self.theta):
            self.circuit.h(i)
            self.circuit.ry(θ, i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Measurement
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> float:
        """
        Execute the circuit with the supplied parameters.
        Args:
            params: Array of shape (n_qubits,)
        Returns:
            Expectation value of the first qubit in the computational basis.
        """
        param_bind = {θ: val for θ, val in zip(self.theta, params)}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, parameter_binds=[param_bind], shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Convert counts to expectation of Z on qubit 0
        exp = 0.0
        for state, cnt in counts.items():
            bit = int(state[::-1][0])  # Qiskit orders bits reversed
            exp += ((-1) ** bit) * cnt
        exp /= self.shots
        return exp


# --------------------------------------------------------------------------- #
# Hybrid layer that bridges PyTorch and Qiskit
# --------------------------------------------------------------------------- #
class HybridQuantumLayer(nn.Module):
    """
    Forward pass runs the quantum circuit; backward uses finite‑difference
    approximation to propagate gradients.
    """

    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2):
        super().__init__()
        self.qc = QuantumExpectationCircuit(n_qubits)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_qubits)
        Returns:
            Tensor of shape (batch, 1) with expectation values.
        """
        # Convert to numpy for circuit execution
        inputs = x.detach().cpu().numpy()
        expectations = np.array([self.qc.run(vec) for vec in inputs])
        return torch.tensor(expectations, dtype=torch.float32, device=x.device).unsqueeze(-1)

    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Finite‑difference gradient approximation.
        """
        eps = 1e-3
        grads = []
        for idx in range(x.shape[1]):
            plus = x.clone()
            minus = x.clone()
            plus[:, idx] += eps
            minus[:, idx] -= eps
            exp_plus = self.forward(plus)
            exp_minus = self.forward(minus)
            grad = (exp_plus - exp_minus) / (2 * eps)
            grads.append(grad)
        grad_inputs = torch.stack(grads, dim=1)
        return grad_inputs * grad_output


# --------------------------------------------------------------------------- #
# End‑to‑end quantum‑enhanced fraud detection model
# --------------------------------------------------------------------------- #
class FraudDetectionHybridQuantum(nn.Module):
    """
    Mirrors FraudDetectionHybrid but replaces the linear classifier with
    the quantum hybrid layer.  The rest of the pipeline remains classical.
    """

    def __init__(
        self,
        input_params,
        layers,
        use_quanvolution: bool = False,
    ) -> None:
        super().__init__()
        # Reuse the classical feature extractor and attention
        self.feature_extractor = build_fraud_detection_program(input_params, layers)
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            self.quanvolution = QuanvolutionFilter()
        # Quantum hybrid head
        self.hybrid = HybridQuantumLayer(n_qubits=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            features = self.quanvolution(x)
        else:
            features = self.feature_extractor(x)
        features = self.attention(features)
        # Quantum head outputs a single expectation; convert to probability
        expectation = self.hybrid(features)
        prob = torch.sigmoid(expectation)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = [
    "QuantumExpectationCircuit",
    "HybridQuantumLayer",
    "FraudDetectionHybridQuantum",
]
