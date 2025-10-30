import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.random import random_circuit

class QuantumCircuitWrapper:
    def __init__(self, n_qubits, backend, shots):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        self.circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas):
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: th} for th in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs.detach().cpu().numpy()
        angles = ctx.inputs
        exp_vals = ctx.circuit.run(angles)
        return torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.inputs
        shift = np.ones_like(inputs) * ctx.shift
        grads = []
        for val, sh in zip(inputs, shift):
            right = ctx.circuit.run([val + sh])
            left = ctx.circuit.run([val - sh])
            grads.append(right - left)
        grads = np.array(grads) * grad_output.cpu().numpy()
        return torch.tensor(grads, dtype=torch.float32, device=grad_output.device), None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits, backend, shots, shift=np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs):
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class QuantumSelfAttention:
    def __init__(self, n_qubits, backend, shots):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

    def _build_circuit(self, rot_vals, ent_vals):
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(rot_vals):
            qc.rx(val, i)
        for i, val in enumerate(ent_vals):
            qc.crx(val, i, i + 1)
        qc.measure_all()
        return qc

    def run(self, inputs):
        outputs = []
        for token in inputs:
            rot_vals = np.full(self.n_qubits, token * np.pi)
            ent_vals = np.full(self.n_qubits - 1, np.sin(token))
            qc = self._build_circuit(rot_vals, ent_vals)
            compiled = transpile(qc, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            counts = job.result().get_counts()
            exp = 0.0
            for bitstring, count in counts.items():
                if bitstring[0] == '1':
                    exp += count
            exp /= self.shots
            outputs.append(1 - 2 * exp)
        return np.array(outputs)

class QuantumQuanvolutionFilter:
    def __init__(self, backend, shots=100, threshold=0.5):
        self.n_qubits = 4
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.thetas = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self.base_circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.base_circuit.rx(self.thetas[i], i)
        rand_circ = random_circuit(self.n_qubits, depth=2, measure=False)
        self.base_circuit += rand_circ
        self.base_circuit.measure_all()

    def forward(self, x):
        bsz = x.size(0)
        patches = []
        for b in range(bsz):
            img = x[b, 0].cpu().numpy()
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = img[r:r+2, c:c+2].flatten()
                    param_binds = {self.thetas[i]: np.pi if val > self.threshold else 0.0
                                   for i, val in enumerate(patch)}
                    compiled = transpile(self.base_circuit, self.backend)
                    qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_binds])
                    job = self.backend.run(qobj)
                    counts = job.result().get_counts()
                    ones = 0
                    total = 0
                    for bitstring, cnt in counts.items():
                        total += cnt
                        ones += bitstring.count('1') * cnt
                    prob = ones / (total * self.n_qubits)
                    patches.append(prob)
        return torch.tensor(patches, dtype=torch.float32).view(bsz, -1)

class UnifiedQuanvolution(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        backend = AerSimulator()
        self.filter = QuantumQuanvolutionFilter(backend, shots=100, threshold=0.5)
        self.attention = QuantumSelfAttention(n_qubits=4, backend=backend, shots=100)
        self.hybrid = Hybrid(n_qubits=1, backend=backend, shots=100, shift=np.pi / 2)
        self.num_classes = num_classes

    def forward(self, x):
        filt_out = self.filter.forward(x)  # (B, 784)
        attn_out = []
        for i in range(filt_out.size(0)):
            tokens = filt_out[i].detach().cpu().numpy()
            attn_vals = self.attention.run(tokens)
            attn_out.append(attn_vals)
        attn_tensor = torch.tensor(attn_out, dtype=torch.float32, device=x.device)
        reduced = attn_tensor.mean(dim=1)
        logits = self.hybrid(reduced)
        if self.num_classes == 2:
            probs = torch.cat([logits, 1 - logits], dim=-1)
            return F.log_softmax(probs, dim=-1)
        else:
            return F.log_softmax(logits, dim=-1)

__all__ = ["UnifiedQuanvolution"]
