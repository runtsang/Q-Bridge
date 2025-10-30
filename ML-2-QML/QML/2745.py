import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """
    Parameterised quantum circuit used for the expectation head.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        if backend is None:
            backend = AerSimulator()
        self.backend = backend
        self.shots = shots
        self._circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys()), dtype=int)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectation = circuit.run(thetas)
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class UnifiedHybridLayer(nn.Module):
    """
    Hybrid dense head that forwards through a parameterised quantum circuit.
    """
    def __init__(self, in_features: int, n_qubits: int = 1,
                 backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.shift = shift
        if backend is None:
            backend = AerSimulator()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = torch.flatten(x, 1)
        probs = HybridFunction.apply(x_flat, self.circuit, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        return self.circuit.run(thetas)
