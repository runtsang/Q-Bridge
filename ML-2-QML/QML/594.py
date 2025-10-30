import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.circuit import Parameter

class QuantumHybridClassifier(nn.Module):
    """
    Quantum‑enhanced classifier that mirrors the classical architecture
    but replaces the final head with a two‑qubit parameter‑shift circuit.
    The circuit is executed on the Aer simulator and wrapped in a
    differentiable torch.autograd.Function for back‑propagation.
    """
    class _QuantumExpectationFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, wrapper, shift):
            # Run the quantum circuit for each logit
            expectations = []
            for logit in logits.squeeze().tolist():
                expectations.append(wrapper.run(logit))
            result = torch.tensor(expectations, dtype=torch.float32, device=logits.device)
            ctx.shift = shift
            ctx.wrapper = wrapper
            ctx.save_for_backward(logits, result)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            logits, _ = ctx.saved_tensors
            shift = ctx.shift
            wrapper = ctx.wrapper
            grads = []
            for logit in logits.squeeze().tolist():
                # Parameter‑shift rule
                exp_plus = wrapper.run(logit + shift)
                exp_minus = wrapper.run(logit - shift)
                grads.append(exp_plus - exp_minus)
            grad_logits = torch.tensor(grads, dtype=torch.float32, device=logits.device) * grad_output
            return grad_logits, None, None

    class _QuantumCircuitWrapper:
        def __init__(self, n_qubits: int, backend):
            self.backend = backend
            self.n_qubits = n_qubits
            self.theta = Parameter('θ')
            self.circuit = qiskit.QuantumCircuit(n_qubits)
            all_qubits = list(range(n_qubits))
            self.circuit.h(all_qubits)
            self.circuit.ry(self.theta, all_qubits)
            self.circuit.measure_all()

        def run(self, theta: float) -> float:
            bound = self.circuit.bind_parameters({self.theta: theta})
            transpiled = transpile(bound, self.backend)
            qobj = assemble(transpiled, shots=200)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            # Expectation of Z on first qubit
            exp = 0.0
            total = 0
            for bitstring, count in counts.items():
                state = int(bitstring[0])  # first qubit
                exp += (1 - 2 * state) * count
                total += count
            return exp / total

    def __init__(self,
                 n_qubits: int = 2,
                 shots: int = 200,
                 shift: float = np.pi / 2,
                 dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(dropout),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84, 1)
        )
        self.backend = Aer.get_backend('aer_simulator')
        self.quantum_wrapper = self._QuantumCircuitWrapper(n_qubits, self.backend)
        self.shift = shift
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        q_out = self._QuantumExpectationFunction.apply(features, self.quantum_wrapper, self.shift)
        return torch.cat((q_out, 1 - q_out), dim=-1)

__all__ = ["QuantumHybridClassifier"]
