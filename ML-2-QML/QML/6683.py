"""Hybrid kernel layer with quantum RBF encoding and a quantum fully‑connected block."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit


class QuantumKernAlAnsatz(tq.QuantumModule):
    """
    Encodes classical data into a quantum state via a list of parameterised gates.
    """

    def __init__(self, gate_specs: List[dict]) -> None:
        super().__init__()
        self.gate_specs = gate_specs

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # initialise with x
        q_device.reset_states(x.shape[0])
        for spec in self.gate_specs:
            params = x[:, spec["input_idx"]]
            func_name_dict[spec["func"]](q_device, wires=spec["wires"], params=params)

        # apply inverse mapping from y
        for spec in reversed(self.gate_specs):
            params = -y[:, spec["input_idx"]]
            func_name_dict[spec["func"]](q_device, wires=spec["wires"], params=params)


class QuantumFullyConnected(tq.QuantumModule):
    """
    Simple parameterised quantum circuit that emulates a fully‑connected layer.
    """
    def __init__(self, n_qubits: int, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    @tq.static_support
    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        thetas_np = thetas.cpu().numpy().flatten()
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: th} for th in thetas_np],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()), dtype=float)
        states = np.array(list(result.keys()), dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return torch.tensor(expectation, device=thetas.device, dtype=thetas.dtype)


class HybridQuantumKernelLayer(tq.QuantumModule):
    """
    Quantum‑classical hybrid kernel layer.

    The kernel is evaluated via a fixed TorchQuantum ansatz.
    The resulting similarity score is fed into a small quantum
    fully‑connected circuit that produces the final output.
    """

    def __init__(self, n_wires: int = 4, shots: int = 1024) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.shots = shots
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        self.kernel_ansatz = QuantumKernAlAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.fc = QuantumFullyConnected(n_qubits=1, shots=self.shots)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel similarity and pass through quantum FC.

        Parameters
        ----------
        x, y : torch.Tensor
            Each of shape (batch, d).  For a single similarity score we
            expect batch of size 1.

        Returns
        -------
        torch.Tensor
            Scalar output of the quantum fully‑connected block.
        """
        # kernel evaluation
        self.kernel_ansatz(self.q_device, x, y)
        kernel_val = torch.abs(self.q_device.states.view(-1)[0])

        # broadcast to FC input
        fc_input = kernel_val.expand(-1, 1)
        return self.fc(fc_input)

    def kernel_matrix(self, a: List[torch.Tensor], b: List[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two lists of tensors."""
        K = np.zeros((len(a), len(b)))
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                K[i, j] = self.forward(xi.unsqueeze(0), yj.unsqueeze(0)).item()
        return K


__all__ = ["HybridQuantumKernelLayer"]
