"""Quantum kernel and variational fully‑connected layer.

The quantum kernel uses a parameterized rotation‑only ansatz
implemented with torchquantum.  The fully‑connected layer is a
single‑qubit qiskit circuit that can be trained by varying a
single rotation angle.  Both components are exposed through the
single class `QuantumKernelMethod` for consistency with the
classical implementation.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import Aer, execute, QuantumCircuit
from qiskit.circuit import Parameter
from typing import Sequence, Iterable

class KernalAnsatz(tq.QuantumModule):
    """Rotation‑only ansatz that encodes two classical vectors."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernelMethod(tq.QuantumModule):
    """
    Combines a quantum RBF‑style kernel with a variational fully‑connected layer.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits for the kernel ansatz.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Variational fully‑connected layer (single‑qubit qiskit circuit)
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.barrier()
        qc.ry(theta, 0)
        qc.measure_all()
        self.fcl_circuit = qc
        self.fcl_backend = Aer.get_backend("qasm_simulator")
        self.fcl_shots = 1000
        self.theta = theta

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel value for two batches."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """Generate the Gram matrix between two collections of tensors."""
        K = torch.stack([self.forward(a_i, torch.stack(b)) for a_i in a])
        return K.detach().cpu().numpy()

    def fcl_run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the variational fully‑connected qiskit circuit."""
        results = []
        for theta_val in thetas:
            bound = {self.theta: theta_val}
            job = execute(self.fcl_circuit,
                          backend=self.fcl_backend,
                          shots=self.fcl_shots,
                          parameter_binds=[bound])
            result = job.result()
            counts = result.get_counts(self.fcl_circuit)
            probs = np.array(list(counts.values())) / self.fcl_shots
            states = np.array(list(counts.keys())).astype(float)
            expectation = np.sum(states * probs)
            results.append(expectation)
        return np.array(results)

__all__ = ["QuantumKernelMethod", "KernalAnsatz"]
