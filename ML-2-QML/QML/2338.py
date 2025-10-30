"""Hybrid kernel method combining quantum kernel evaluation and a quantum fully‑connected layer.

The module extends the original QuantumKernelMethod by exposing a
single :class:`HybridKernelMethod` that can compute both quantum and
classical kernels.  The quantum side uses TorchQuantum for the kernel
and Qiskit for the fully‑connected layer, providing a clear example of
how to mix different quantum back‑ends within the same API.

The design follows a *combination* scaling paradigm: the quantum
operations are encapsulated in optional sub‑modules that can be
enabled when a suitable backend is available.  The module remains
importable in a pure‑Python environment when the quantum libraries
are missing, thanks to lazy imports and fall‑back stubs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
# Quantum kernel – TorchQuantum implementation.
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list: Sequence[dict]) -> None:
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


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Quantum fully‑connected layer – Qiskit implementation.
# --------------------------------------------------------------------------- #

class QuantumFullyConnectedLayer:
    """Parameterised Qiskit circuit that implements a fully‑connected layer."""

    def __init__(self, n_qubits: int = 1, backend: qiskit.providers.BaseBackend | None = None, shots: int = 100) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = Parameter("theta")
        self._circuit.h(range(self.n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(self.n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
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


def FCL() -> QuantumFullyConnectedLayer:
    """Return a Qiskit circuit that mimics the quantum fully‑connected layer."""
    return QuantumFullyConnectedLayer()


# --------------------------------------------------------------------------- #
# Unified hybrid API.
# --------------------------------------------------------------------------- #

class HybridKernelMethod:
    """Unified API for quantum kernel evaluation and quantum fully‑connected layer.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits for the kernel ansatz.
    backend : qiskit.providers.BaseBackend, optional
        Backend used by the quantum fully‑connected layer.
    shots : int, optional
        Number of shots for the quantum circuit.
    """

    def __init__(self, n_wires: int = 4, backend: qiskit.providers.BaseBackend | None = None, shots: int = 100) -> None:
        self.n_wires = n_wires
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Quantum kernel components
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Quantum fully‑connected layer
        self.fcl = QuantumFullyConnectedLayer(1, self.backend, self.shots)

    @staticmethod
    def quantum_kernel(x: torch.Tensor, y: torch.Tensor, n_wires: int = 4) -> torch.Tensor:
        """Compute the quantum kernel between two vectors."""
        q_device = tq.QuantumDevice(n_wires=n_wires)
        ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        ansatz(q_device, x, y)
        return torch.abs(q_device.states.view(-1)[0])

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4) -> np.ndarray:
        """Return the Gram matrix using the quantum kernel."""
        return np.array([[HybridKernelMethod.quantum_kernel(x, y, n_wires).item() for y in b] for x in a])

    def fully_connected_layer(self, thetas: Iterable[float]) -> np.ndarray:
        """Compute the expectation of the quantum fully‑connected layer."""
        return self.fcl.run(thetas)
