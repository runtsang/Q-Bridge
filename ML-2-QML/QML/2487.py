import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

# QCNN-inspired quantum primitives
def conv_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi/2, 0)
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

class QuantumHybridCircuit:
    """Two‑qubit circuit that emulates a QCNN convolution followed by a pooling step."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100, shift: float = np.pi / 2) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = backend or AerSimulator()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each theta and return expectation of Z on qubit 0."""
        expectations = []
        for theta in thetas:
            qc = QuantumCircuit(self.n_qubits)
            # Convolution step
            qc.append(conv_circuit([theta, theta, theta]), range(self.n_qubits))
            qc.barrier()
            # Pooling step
            qc.append(pool_circuit([theta, theta, theta]), range(self.n_qubits))
            qc.barrier()
            qc.measure_all()
            compiled = transpile(qc, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            expectation = 0.0
            for bitstring, count in counts.items():
                first_qubit = int(bitstring[0])
                prob = count / self.shots
                expectation += (1 if first_qubit == 0 else -1) * prob
            expectations.append(expectation)
        return np.array(expectations)

class HybridFunction(autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumHybridCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        expectations = circuit.run(thetas)
        result = torch.tensor(expectations, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        shift = np.ones_like(thetas) * ctx.shift
        gradients = []
        for idx, theta in enumerate(thetas):
            right = ctx.circuit.run([theta + shift[idx]])[0]
            left = ctx.circuit.run([theta - shift[idx]])[0]
            gradients.append(right - left)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None

class HybridQuantumLayer(nn.Module):
    """Quantum layer that forwards activations through a parameterised circuit."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumHybridCircuit(n_qubits, backend, shots, shift)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class HybridQCNNNet(nn.Module):
    """Quantum‑only network that uses a QCNN‑inspired circuit for classification."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_head = HybridQuantumLayer(n_qubits, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.quantum_head(inputs)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridCircuit", "HybridFunction", "HybridQuantumLayer", "HybridQCNNNet"]
