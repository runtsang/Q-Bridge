import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import qiskit
from qiskit import QuantumCircuit as _QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumExpectation(Function):
    """Differentiable quantum expectation via shift rule."""
    @staticmethod
    def forward(ctx, angles: torch.Tensor, circuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        ang_np = angles.detach().cpu().numpy()
        expectations = circuit.run(ang_np)
        ctx.save_for_backward(angles)
        return torch.tensor(expectations, dtype=torch.float32, device=angles.device)

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        ang_np = angles.detach().cpu().numpy()
        gradients = []
        for ang in ang_np:
            f_plus = circuit.run([ang + shift])[0]
            f_minus = circuit.run([ang - shift])[0]
            grad = (f_plus - f_minus) / (2 * np.sin(shift))
            gradients.append(grad)
        gradients = torch.tensor(gradients, dtype=torch.float32, device=angles.device)
        return grad_output * gradients, None, None

class QuantumCircuitWrapper:
    """Wrapper for executing a parameterized Qiskit circuit."""
    def __init__(self, circuit: _QC, backend, shots: int):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.transpiled = None

    def run(self, angles: np.ndarray) -> np.ndarray:
        if self.transpiled is None:
            self.transpiled = transpile(self.circuit, self.backend)
        if angles.ndim > 1:
            angles = angles.reshape(-1)
        expectations = []
        for ang in angles:
            param_binds = [{self.circuit.parameters[0]: ang}]
            qobj = assemble(self.transpiled, shots=self.shots, parameter_binds=param_binds)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            probs = np.array(list(counts.values())) / self.shots
            expectation = np.sum(states * probs)
            expectations.append(expectation)
        return np.array(expectations)

class HybridQuantumFullyConnectedLayer(nn.Module):
    """Classical dense layer followed by a parameterized quantum circuit head."""
    def __init__(self, in_features: int, n_qubits: int = 1,
                 backend=None, shots: int = 100, shift: float = np.pi/2):
        super().__init__()
        self.linear = nn.Linear(in_features, n_qubits)
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.shift = shift
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = _QC(self.n_qubits)
        all_qubits = list(range(self.n_qubits))
        theta = qiskit.circuit.Parameter("theta")
        qc.h(all_qubits)
        qc.barrier()
        qc.ry(theta, all_qubits)
        qc.measure_all()
        return QuantumCircuitWrapper(qc, self.backend, self.shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        angles = linear_out.squeeze(-1)
        return QuantumExpectation.apply(angles, self.circuit, self.shift)
