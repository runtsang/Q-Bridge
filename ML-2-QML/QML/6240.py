import torch
import numpy as np
import qiskit
from qiskit import assemble, transpile
import ml_code

class VariationalThreeQubitCircuit:
    """3‑qubit variational circuit with Ry rotations and an entangling CNOT chain."""
    def __init__(self, backend, shots: int):
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.ParameterVector("theta", 3)
        self.circuit = self._build()

    def _build(self):
        qc = qiskit.QuantumCircuit(3)
        for i, p in enumerate(self.theta):
            qc.ry(p, i)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        return qc

    def run(self, params: np.ndarray) -> float:
        bind = {self.theta[i]: params[i] for i in range(3)}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        shots = self.shots
        # Compute expectation of Z on qubit 0
        p1 = sum(count for bitstring, count in result.items() if bitstring[-1] == '1')
        p1 = p1 / shots
        expectation = 1 - 2 * p1
        return expectation

class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable bridge that evaluates the variational circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalThreeQubitCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        input_value = inputs.item()
        params = np.array([input_value] * 3)
        expectation = circuit.run(params)
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        input_value = inputs.item()
        params_plus = np.array([input_value + shift] * 3)
        params_minus = np.array([input_value - shift] * 3)
        exp_plus = ctx.circuit.run(params_plus)
        exp_minus = ctx.circuit.run(params_minus)
        grad = (exp_plus - exp_minus) / (2 * shift)
        grad *= 3  # 3 identical parameters
        return grad * grad_output, None, None

class HybridQuantum(ml_code.Hybrid):
    """Hybrid layer that forwards activations through the 3‑qubit variational circuit."""
    def __init__(self, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = VariationalThreeQubitCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.circuit, self.shift)

class QCNetQuantum(ml_code.QCNet):
    """Same architecture as QCNet but with a quantum hybrid head."""
    def __init__(self, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = HybridQuantum(backend, shots, shift)

__all__ = ["VariationalThreeQubitCircuit", "QuantumHybridFunction", "HybridQuantum", "QCNetQuantum"]
