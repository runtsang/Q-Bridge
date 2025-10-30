import torch
import torchquantum as tq
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class QuantumKernel(tq.QuantumModule):
    """
    A TorchQuantum ansatz that encodes two‑dimensional classical data
    into a 4‑wire quantum state and returns the overlap amplitude.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule(
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

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute the full Gram matrix for the quantum kernel."""
        return np.array([[self(x, y).item() for y in b] for x in a])

class QuantumSampler(tq.QuantumModule):
    """
    Variational sampler built with Qiskit that outputs a probability
    distribution over two qubits. The circuit is parameterized by
    both data and trainable weights.
    """
    def __init__(self):
        super().__init__()
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler
        )

    def forward(self, data: torch.Tensor):
        """Return the sampler output for the given data."""
        return self.qnn(data)

class HybridKernelSampler(tq.QuantumModule):
    """
    Bundles the quantum kernel and the sampler QNN, exposing a
    convenient API for hybrid workflows.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.kernel = QuantumKernel(n_wires)
        self.sampler = QuantumSampler()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        return self.kernel.kernel_matrix(a, b)

    def sampler_output(self, data: torch.Tensor):
        return self.sampler(data)

__all__ = ["HybridKernelSampler"]
