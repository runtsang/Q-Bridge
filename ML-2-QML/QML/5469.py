import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer, transpile, assemble
import torch
import torch.nn as nn
from torch.autograd import Function

class QuantumAttentionCircuit:
    """
    Parametric rotation‑entangle circuit used as a quantum self‑attention block.
    """
    def __init__(self, n_qubits=4, backend=None, shots=1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f'theta{i}') for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, params):
        param_dict = {theta: val for theta, val in zip(self.theta, params)}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_dict])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        exp = 0.0
        for outcome, count in result.items():
            bit = int(outcome[0])
            exp += (1 - 2 * bit) * count
        return exp / self.shots

class QuantumAttentionFunction(Function):
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs
        batch = inputs.shape[0]
        outputs = []
        for i in range(batch):
            params = inputs[i].cpu().numpy()
            exp = ctx.circuit.run(params)
            outputs.append(exp)
        return torch.tensor(outputs, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        shift = ctx.shift
        grad = []
        for i in range(grad_output.shape[0]):
            inp = ctx.inputs[i].cpu().numpy()
            up = ctx.circuit.run(inp + shift)
            down = ctx.circuit.run(inp - shift)
            grad.append((up - down) / 2)
        return torch.tensor(grad, dtype=grad_output.dtype, device=grad_output.device), None, None

class HybridQuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, shift=np.pi/2, backend=None, shots=1024):
        super().__init__()
        self.circuit = QuantumAttentionCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x):
        return QuantumAttentionFunction.apply(x, self.circuit, self.shift)

class QuantumNATHybridQuantumModule(nn.Module):
    """
    Quantum module that can be injected into the classical core.
    It implements a self‑attention style circuit followed by a
    quantum expectation head.
    """
    def __init__(self, embed_dim=4, n_qubits=4, shift=np.pi/2):
        super().__init__()
        self.quantum_layer = HybridQuantumLayer(n_qubits, shift)

    def forward(self, x):
        return self.quantum_layer(x)
