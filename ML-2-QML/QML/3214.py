"""Hybrid classical‑quantum kernel binary classifier (quantum version).

This module mirrors the classical implementation but replaces the RBF
kernel with a quantum kernel evaluated by a fixed TorchQuantum ansatz.
The backbone is identical to the classical version; the hybrid head
consists of a quantum expectation layer and a quantum‑kernel‑derived
similarity score that together drive the final decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Quantum circuit for the hybrid head
class QuantumCircuit:
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# Differentiable interface between PyTorch and the quantum circuit
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.circuit.run([value + shift[idx]])
            expectation_left = ctx.circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

# Quantum kernel using TorchQuantum
class KernalAnsatz(tq.QuantumModule):
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

# Sigmoid activation with optional shift
class ShiftedSigmoid(nn.Module):
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class HybridQuantumKernelNet(nn.Module):
    """CNN backbone + hybrid head using quantum expectation and quantum kernel."""

    def __init__(
        self,
        num_prototypes: int = 8,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        # Backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_circuit = QuantumCircuit(1, backend, shots=100)

        # Quantum kernel
        self.kernel = Kernel()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, self.fc3.out_features))
        self.kernel_linear = nn.Linear(num_prototypes, 1)

        # Final sigmoid
        self.shifted_sigmoid = ShiftedSigmoid(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (batch, 1)

        # Quantum expectation output
        expectation = HybridFunction.apply(x.squeeze(-1), self.quantum_circuit, self.shifted_sigmoid.shift)  # (1, batch)
        expectation = expectation.T  # (batch, 1)

        # Quantum kernel similarities
        feat = x.squeeze(-1)  # (batch,)
        sims = torch.stack([self.kernel(feat, p) for p in self.prototypes], dim=1).squeeze(-1)  # (batch, num_prototypes)
        kernel_logits = self.kernel_linear(sims)  # (batch, 1)

        # Combine logits
        logits = expectation + kernel_logits  # (batch, 1)
        probs = self.shifted_sigmoid(logits)

        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumKernelNet"]
