import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector


class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter to keep the architecture identical."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuantumCircuitWrapper:
    """Parametrised one‑qubit circuit for the quantum expectation head."""
    def __init__(self, n_qubits: int = 1, shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.backend = AerSimulator()
        self.shots = shots

        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
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
            probs = counts / self.shots
            return np.sum(states * probs)

        return np.array([expectation(result)])


class HybridFunctionQuantum(torch.autograd.Function):
    """Quantum‑enabled differentiable head using the wrapper circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.save_for_backward(inputs)
        inputs_np = inputs.detach().cpu().numpy()
        expectations = circuit.run(inputs_np)
        return torch.tensor(expectations, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        inputs_np = inputs.detach().cpu().numpy()
        grad_inputs = []
        for val in inputs_np:
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grad = (right - left) / (2 * shift) if shift!= 0 else 0
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32)
        return grad_inputs * grad_output, None, None


class SamplerQNN(nn.Module):
    """Sampler‑based quantum head producing a two‑class probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.backend = AerSimulator()
        self.shots = 1024
        self.input_params = ParameterVector("input", 1)
        self.weight_params = ParameterVector("weight", 1)

        self.circuit = QuantumCircuit(1)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.measure_all()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_list = inputs.detach().cpu().numpy()
        probs = []
        for val in input_list if input_list.ndim > 1 else [input_list]:
            bind_dict = {
                self.input_params[0]: val,
                self.weight_params[0]: 0.5,  # fixed weight for demonstrative purposes
            }
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind_dict])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            prob0 = result.get("0", 0) / self.shots
            prob1 = result.get("1", 0) / self.shots
            probs.append([prob1, prob0])  # class1, class0
        return torch.tensor(probs, dtype=torch.float32)


class HybridBinaryClassifier(nn.Module):
    """
    Hybrid binary classifier that can use either a quantum expectation head
    or a sampler‑based head.  The architecture mirrors the classical version
    but replaces the final activation with a quantum operation.
    """
    def __init__(self, head: str = "quantum_expectation", shift: float = np.pi / 4) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc1 = nn.Linear(4 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.shift = shift

        if head == "quantum_expectation":
            self.quantum_circuit = QuantumCircuitWrapper(n_qubits=1, shots=100)
            self.head = HybridFunctionQuantum.apply
        elif head == "sampler":
            self.sampler = SamplerQNN()
            self.head = self.sampler.forward
        else:
            raise ValueError("head must be 'quantum_expectation' or'sampler'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qfilter(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.head == self.sampler.forward:
            probs = self.sampler(x)
        else:
            probs = self.head(x, self.quantum_circuit, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier"]
