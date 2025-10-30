import qiskit
import numpy as np
import torch
import torch.nn as nn

class QuantumCircuit:
    """Two‑qubit variational circuit with a single Ry rotation on each qubit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        # Simple ansatz: H on all, then Ry(θ) on each qubit
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each θ in thetas and return Z‑expectation values."""
        compiled = qiskit.transpile(self.circuit, self.backend)
        job = self.backend.run(qiskit.assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas]
        ))
        result = job.result().get_counts()
        if isinstance(result, list):
            return np.array([self._expectation(c) for c in result])
        else:
            return np.array([self._expectation(result)])

    def _expectation(self, count_dict: dict) -> float:
        """Expectation of Z on the first qubit."""
        counts = np.array(list(count_dict.values()))
        states = np.array([int(s[-1]) for s in count_dict.keys()])  # last bit is first qubit
        probs = counts / self.shots
        return np.sum((2 * states - 1) * probs)

class HybridFunction(torch.autograd.Function):
    """Forward and backward hooks that run the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectation = circuit.run(thetas)
        result = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run(np.array([val + shift]))
            left = ctx.circuit.run(np.array([val - shift]))
            grad = (right - left) / 2
            grads.append(grad)
        grad_input = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grad_input * grad_output, None, None

class Hybrid(nn.Module):
    """Module that forwards a scalar through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots, shift)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)
