"""Hybrid quantum kernel combining a data‑encoding ansatz with a
parameterised Qiskit circuit that mimics a fully‑connected layer.

The module provides a TorchQuantum implementation of the kernel and
a Qiskit wrapper that evaluates an expectation value based on the
kernel output.  The public API matches the classical version so that
users can swap back‑ends without changing the surrounding code.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----------------------------------------------------------------------
# 1. Quantum kernel ansatz
# ----------------------------------------------------------------------
class QuantumAnsatz(tq.QuantumModule):
    """Encodes two classical vectors into a single quantum state.
    The encoding uses a list of parameterised single‑qubit rotations
    followed by a reverse‑order un‑encoding that mirrors the classical
    kernel construction.
    """
    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self,
                q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        # Reset device to the batch size of the first vector
        q_device.reset_states(x.shape[0])
        # Forward encoding of x
        for info in self.func_list:
            params = x[:, info["input_idx"]] \
                if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device,
                                         wires=info["wires"],
                                         params=params)
        # Reverse‑order un‑encoding of y (with negated parameters)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] \
                if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device,
                                         wires=info["wires"],
                                         params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap between two data
    vectors using the :class:`QuantumAnsatz` defined above.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the absolute value of the first amplitude
        return torch.abs(self.q_device.states.view(-1)[0])

# ----------------------------------------------------------------------
# 2. Quantum fully‑connected layer (Qiskit)
# ----------------------------------------------------------------------
import qiskit

class QuantumFCL:
    """A Qiskit‑based parameterised circuit that serves as a
    quantum analogue of a fully‑connected layer.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Sequence[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

# ----------------------------------------------------------------------
# 3. Hybrid quantum kernel
# ----------------------------------------------------------------------
class HybridKernel(tq.QuantumModule):
    """Quantum hybrid kernel that combines the quantum kernel and the
    Qiskit fully‑connected layer.  The kernel value is fed as a
    parameter to the Qiskit circuit, mimicking the classical
    ``HybridKernel`` that applies a fully‑connected layer after the
    RBF transformation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.kernel = QuantumKernel()
        self.fcl = QuantumFCL()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Evaluate quantum kernel
        k = self.kernel(x, y).item()
        # Use the kernel value as the theta for the FCL circuit
        expectation = self.fcl.run([k])[0]
        return torch.tensor(expectation, dtype=torch.float32)

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """Return the Gram matrix between two collections of feature vectors
    using :class:`HybridKernel`.
    """
    kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumAnsatz",
           "QuantumKernel",
           "QuantumFCL",
           "HybridKernel",
           "kernel_matrix"]
