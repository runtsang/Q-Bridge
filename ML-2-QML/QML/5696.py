import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumKernel:
    def __init__(self, n_qubits=4, shots=1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.base_circuit = QuantumCircuit(n_qubits)
        for _ in range(8):
            for q in range(n_qubits):
                self.base_circuit.ry(np.random.rand() * 2 * np.pi, q)
            for q in range(n_qubits - 1):
                self.base_circuit.cx(q, q + 1)
        self.base_circuit.measure_all()

    def run(self, inputs):
        results = []
        for inp in inputs:
            circ = self.base_circuit.copy()
            for q, val in enumerate(inp):
                circ.ry(val, q)
            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = []
            for q in range(self.n_qubits):
                z_exp = 0.0
                for bitstring, count in counts.items():
                    bit = bitstring[::-1][q]
                    z = 1.0 if bit == '0' else -1.0
                    z_exp += z * count
                z_exp /= self.shots
                exp.append(z_exp)
            results.append(exp)
        return np.array(results)

class QuantumExpectationHead:
    def __init__(self, n_qubits=1, shots=1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas):
        results = []
        for theta in thetas:
            circ = self.circuit.copy()
            circ.ry(theta, 0)
            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            z_exp = 0.0
            for bitstring, count in counts.items():
                bit = bitstring[::-1][0]
                z = 1.0 if bit == '0' else -1.0
                z_exp += z * count
            z_exp /= self.shots
            results.append(z_exp)
        return np.array(results)

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumExpectationHead, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().flatten()
        expectations = circuit.run(thetas)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for inp in inputs.cpu().numpy():
            right = ctx.circuit.run([inp + shift])
            left = ctx.circuit.run([inp - shift])
            grad = (right - left) / 2.0
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits=1, shift=np.pi/2, shots=1024):
        super().__init__()
        self.circuit = QuantumExpectationHead(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor):
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class QuanvolutionFilter(nn.Module):
    def __init__(self, n_qubits=4, shots=1024):
        super().__init__()
        self.kernel = QuantumKernel(n_qubits, shots)

    def forward(self, x: torch.Tensor):
        bsz = x.shape[0]
        h, w = x.shape[2], x.shape[3]
        patches = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                patch = x[:, :, r:r+2, c:c+2]
                patch = patch.squeeze(1)
                patch_flat = patch.reshape(bsz, -1)
                outputs = self.kernel.run(patch_flat)
                patches.append(torch.tensor(outputs, dtype=x.dtype, device=x.device))
        return torch.cat(patches, dim=1)

class QuanvolutionHybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.qfilter = QuanvolutionFilter()
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits=1, shift=np.pi/2, shots=1024)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.qfilter(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)
