from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.random import random_circuit
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuanvCircuit:
    """Quantum convolutional filter that emulates a classical 2‑D kernel."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        """Return the mean probability of measuring |1> across all qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return np.full((self.n_qubits,), counts / (self.shots * self.n_qubits))

class KernalAnsatz(tq.QuantumModule):
    """TorchQuantum ansatz that encodes two vectors and returns their overlap."""
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

class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded vectors."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
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

class ConvKernelQuantum:
    """Hybrid quantum filter that combines a QuanvCircuit with a TorchQuantum kernel."""
    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        backend = Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(kernel_size, backend, shots, threshold)
        self.kernel = Kernel()

    def run(self, data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data : np.ndarray
            2‑D input array of shape (kernel_size, kernel_size).
        ref_data : np.ndarray
            Reference 2‑D array for kernel similarity.

        Returns
        -------
        np.ndarray
            Combined output: mean conv probability multiplied by quantum kernel similarity.
        """
        conv_out = self.conv.run(data).mean()
        ref_conv_out = self.conv.run(ref_data).mean()

        # Convert to torch tensors for kernel evaluation
        conv_tensor = torch.tensor(conv_out, dtype=torch.float32).unsqueeze(0)
        ref_tensor = torch.tensor(ref_conv_out, dtype=torch.float32).unsqueeze(0)

        kernel_sim = self.kernel(conv_tensor, ref_tensor).item()
        return conv_out * kernel_sim

def Conv() -> ConvKernelQuantum:
    """Return a quantum hybrid filter compatible with the original Conv name."""
    return ConvKernelQuantum()

__all__ = ["Conv"]
