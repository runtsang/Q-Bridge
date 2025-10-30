from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import Aer, execute


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data via a programmable list of quantum gates."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel that augments a classical‑style ansatz with a
    parameterised fully‑connected quantum layer (implemented with
    a one‑qubit qiskit circuit).  The final kernel value is the
    product of the ansatz kernel and the FCL expectation, giving a
    hybrid scaling that leverages both quantum and classical
    expressivity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Classical‑style ansatz
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Parameterised fully‑connected quantum layer (1‑qubit)
        self.fcl_backend = Aer.get_backend("qasm_simulator")
        self.fcl_circuit = self._build_fcl_circuit()

    def _build_fcl_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        qc.h(0)
        qc.barrier()
        qc.ry(theta, 0)
        qc.measure_all()
        return qc

    def _fcl_expectation(self, theta: float) -> float:
        bound = self.fcl_circuit.bind_parameters({self.fcl_circuit.parameters[0]: theta})
        job = execute(bound, self.fcl_backend, shots=100)
        result = job.result().get_counts(bound)
        probs = np.array(list(result.values())) / 100
        states = np.array(list(result.keys()), dtype=float)
        return np.sum(states * probs)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return a hybrid Gram matrix combining ansatz kernel and FCL expectation."""
        mat = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                # Ansatz kernel
                self.forward(self.q_device, x.reshape(1, -1), y.reshape(1, -1))
                ansatz_val = torch.abs(self.q_device.states.view(-1)[0]).item()

                # FCL expectation (use first component as theta)
                theta_x = x[0].item()
                theta_y = y[0].item()
                fcl_val = self._fcl_expectation(theta_x) * self._fcl_expectation(theta_y)

                mat[i, j] = ansatz_val * fcl_val
        return mat


__all__ = ["QuantumKernelMethod"]
