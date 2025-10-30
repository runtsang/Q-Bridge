"""Hybrid quantum‑classical estimator using Qiskit’s EstimatorQNN."""
from __future__ import annotations

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit.quantum_info import Pauli

class HybridEstimatorQNN(torch.nn.Module):
    """
    Variational quantum neural network that outputs a scalar regression value.
    The circuit encodes two classical inputs and two variational angles, and
    measures the expectation value of Pauli‑Z on a single qubit.
    """
    def __init__(self) -> None:
        super().__init__()
        # Define parameter registers
        input_params = [Parameter("x0"), Parameter("x1")]
        weight_params = [Parameter("w0"), Parameter("w1")]

        # Build a compact variational circuit
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_params[0], 0)
        qc.rz(weight_params[0], 0)
        qc.ry(input_params[1], 0)
        qc.rz(weight_params[1], 0)

        # Observable and estimator
        observable = Pauli("Z")
        estimator = Estimator()

        # Instantiate the Qiskit EstimatorQNN layer
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, 2)

        Returns
        -------
        torch.Tensor of shape (batch,)
            Scalar regression predictions from the quantum circuit.
        """
        # Convert to numpy for the estimator
        inputs_np = inputs.detach().cpu().numpy()
        # Evaluate the variational circuit
        preds = self.estimator_qnn.predict(inputs_np)
        return torch.tensor(preds, dtype=inputs.dtype, device=inputs.device).squeeze(-1)

__all__ = ["HybridEstimatorQNN"]
