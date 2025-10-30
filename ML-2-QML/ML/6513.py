import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumExpectationFunction(torch.autograd.Function):
    """Custom autograd function that evaluates the quantum expectation value
    for a batch of inputs and provides a finite‑difference gradient."""
    @staticmethod
    def forward(ctx, data, qc, shift):
        data_np = data.detach().cpu().numpy()
        expectations = []
        for sample in data_np:
            flat = sample.flatten()[:qc.n_qubits]
            patch = flat.reshape(int(np.sqrt(qc.n_qubits)), int(np.sqrt(qc.n_qubits)))
            exp = qc.run(patch)
            expectations.append(exp)
        expectations = torch.tensor(expectations, device=data.device, dtype=data.dtype)
        ctx.save_for_backward(data)
        ctx.qc = qc
        ctx.shift = shift
        return expectations

    @staticmethod
    def backward(ctx, grad_output):
        data, = ctx.saved_tensors
        qc = ctx.qc
        shift = ctx.shift
        grad_input = torch.zeros_like(data)
        data_np = data.detach().cpu().numpy()
        for idx in np.ndindex(*data_np.shape):
            orig = data_np[idx]
            plus = orig + shift
            minus = orig - shift
            data_plus = data_np.copy()
            data_minus = data_np.copy()
            data_plus[idx] = plus
            data_minus[idx] = minus
            patch_plus = data_plus.reshape(-1)[:qc.n_qubits].reshape(int(np.sqrt(qc.n_qubits)), int(np.sqrt(qc.n_qubits)))
            patch_minus = data_minus.reshape(-1)[:qc.n_qubits].reshape(int(np.sqrt(qc.n_qubits)), int(np.sqrt(qc.n_qubits)))
            exp_plus = qc.run(patch_plus)
            exp_minus = qc.run(patch_minus)
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_input[idx] = grad * grad_output[0]
        return grad_input, None, None

class ConvHybridFilter(nn.Module):
    """Hybrid convolutional filter that combines a classical convolution
    with a quantum expectation layer.  It is drop‑in compatible with
    the original Conv.py interface."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 quantum_backend=None,
                 shots: int = 100,
                 shift: float = 0.01):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.quantum_backend = quantum_backend or AerSimulator()
        self.shots = shots
        self.shift = shift
        self.n_qubits = kernel_size ** 2
        self._build_quantum_circuit()

    def _build_quantum_circuit(self):
        self.qc = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self.theta = theta
        self.qc.h(range(self.n_qubits))
        for i in range(self.n_qubits):
            self.qc.rx(theta[i], i)
        for i in range(self.n_qubits - 1):
            self.qc.cx(i, i + 1)
        self.qc.measure_all()
        self.compiled = transpile(self.qc, self.quantum_backend)

    def _quantum_expectation(self, data: np.ndarray) -> float:
        flat = data.flatten()
        param_binds = [{self.theta[i]: np.pi if val > self.threshold else 0}
                       for i, val in enumerate(flat)]
        qobj = assemble(self.compiled, shots=self.shots,
                        parameter_binds=param_binds)
        job = self.quantum_backend.run(qobj)
        result = job.result()
        counts = result.get_counts(self.qc)
        exp_z = 0.0
        for bitstr, cnt in counts.items():
            z = 1 if bitstr[-1] == '1' else -1
            exp_z += z * cnt
        exp_z /= self.shots
        return exp_z

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        classical_mean = activations.mean()
        quantum_exp = QuantumExpectationFunction.apply(data, self.qc, self.shift)
        return 0.5 * classical_mean + 0.5 * quantum_exp

    def run(self, data):
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.forward(tensor).item()

def Conv(kernel_size: int = 2,
         threshold: float = 0.0,
         quantum_backend=None,
         shots: int = 100,
         shift: float = 0.01):
    return ConvHybridFilter(kernel_size=kernel_size,
                            threshold=threshold,
                            quantum_backend=quantum_backend,
                            shots=shots,
                            shift=shift)
