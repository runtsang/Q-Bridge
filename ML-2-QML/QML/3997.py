from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

class QuantumEstimator(nn.Module):
    """
    Variational quantum circuit that mirrors the small EstimatorQNN example.
    It evaluates the expectation value of a single Y Pauli observable.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_param = Parameter("input1")
        self.weight_param = Parameter("weight1")

        # Simple 1‑qubit circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        self.observable = SparsePauliOp.from_list([("Y", 1)])
        self.backend = Aer.get_backend("statevector_simulator")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the circuit for each sample in the batch.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (..., 2).  The first column is bound to
            ``input_param`` and the second to ``weight_param``.

        Returns
        -------
        torch.Tensor
            Expectation values of shape (..., 1).
        """
        # Extract parameters
        input_vals = x[..., 0].detach().cpu().numpy()
        weight_vals = x[..., 1].detach().cpu().numpy()

        # Evaluate each sample
        exp_vals = []
        for inp, w in zip(input_vals, weight_vals):
            bound = self.circuit.bind_parameters({
                self.input_param: inp,
                self.weight_param: w
            })
            result = execute(bound, self.backend, shots=1).result()
            state = result.get_statevector(bound)
            exp = np.vdot(state, self.observable.data[0] @ state).real
            exp_vals.append(exp)

        return torch.tensor(exp_vals, dtype=torch.float32).unsqueeze(-1)

__all__ = ["QuantumEstimator"]
