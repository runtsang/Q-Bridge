"""Quantum counterpart of FraudDetectionHybrid built with TorchQuantum.

This module exposes the same functional API as the classical variant
while leveraging a variational quantum kernel and a lightweight
EstimatorQNN‑style circuit.  The design mirrors the classical
pipeline: kernel evaluation → regression head.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import QuantumModule

class FraudDetectionHybridQ(QuantumModule):
    """
    Quantum‑classical hybrid model:
    * `kernel_ansatz` – a list of rotation gates that encode data.
    * `regression_circuit` – a tiny variational circuit that acts as the
      EstimatorQNN head.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Kernel ansatz: one Ry gate per wire, parameterised by input data.
        self.kernel_ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
        ]

        # Regression circuit – a single qubit with a parameterised rotation
        # and a measurement that is weighted by the kernel value.
        self.regression_circuit = tq.QuantumCircuit(self.n_wires)
        self.regression_circuit.h(0)
        self.regression_circuit.ry(tq.Param("x"), 0)
        self.regression_circuit.rx(tq.Param("w"), 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum kernel and return a scalar regression output.
        """
        # 1️⃣ Kernel evaluation
        self.q_device.reset_states(x.shape[0])
        for gate in self.kernel_ansatz:
            params = x[:, gate["input_idx"]] if tq.op_name_dict[gate["func"]].num_params else None
            func_name_dict[gate["func"]](self.q_device, wires=gate["wires"], params=params)
        for gate in reversed(self.kernel_ansatz):
            params = -y[:, gate["input_idx"]] if tq.op_name_dict[gate["func"]].num_params else None
            func_name_dict[gate["func"]](self.q_device, wires=gate["wires"], params=params)

        kernel_val = torch.abs(self.q_device.states.view(-1)[0])

        # 2️⃣ Regression head
        pred = self.regression_circuit.forward(x)
        output = pred @ torch.tensor([kernel_val], device=x.device)
        return output

__all__ = ["FraudDetectionHybridQ"]
