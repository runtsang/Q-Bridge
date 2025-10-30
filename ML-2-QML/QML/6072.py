import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Sequence

class QuantumKernelAnsatz(tq.QuantumModule):
    """Quantum data‑encoding ansatz based on a list of gates."""
    def __init__(self, gate_list):
        super().__init__()
        self.gate_list = gate_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.gate_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.gate_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel using a parameter‑free ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuantumClassifierCircuit:
    """Variational circuit for classification mirroring the classical build function."""
    def __init__(self, num_qubits: int, depth: int):
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)
        self.circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(self.encoding, range(num_qubits)):
            self.circuit.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

    def get_encoding(self):
        return list(self.encoding)

    def get_weights(self):
        return list(self.weights)

    def get_observables(self):
        return self.observables

class QuantumKernelClassifier:
    """Hybrid quantum kernel and variational classifier."""
    def __init__(self, n_wires: int = 4, depth: int = 3):
        self.kernel = QuantumKernel(n_wires)
        self.classifier = QuantumClassifierCircuit(n_wires, depth)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder: return zero logits; in practice bind parameters and simulate
        return torch.zeros(x.shape[0], 2)

__all__ = ["QuantumKernelAnsatz", "QuantumKernel", "kernel_matrix", "QuantumClassifierCircuit", "QuantumKernelClassifier"]
