import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile, Aer
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumCircuit:
    """Two‑qubit variational circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
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
    """Differentiable interface between PyTorch and a Qiskit quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        grads = []
        for val in inputs.tolist():
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class QuantumTransformerBlock(tq.QuantumModule):
    """Single‑layer quantum transformer that processes a sequence of embeddings."""
    def __init__(self, embed_dim: int, n_qubits: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i % n_qubits]}
                for i in range(embed_dim)
            ]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)

class HybridNet(nn.Module):
    """CNN + optional quantum transformer + quantum expectation head."""
    def __init__(
        self,
        num_classes: int = 1,
        use_transformer: bool = False,
        transformer_layers: int = 2,
        embed_dim: int = 64,
        n_qubits_transformer: int = 8,
        n_qubits_head: int = 2,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        # CNN backbone identical to ML version
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, embed_dim)
        self.use_transformer = use_transformer
        if use_transformer:
            self.q_transformer = QuantumTransformerBlock(embed_dim, n_qubits_transformer)
            self.q_device = tq.QuantumDevice(n_wires=n_qubits_transformer)
        # Linear projection to a single parameter for the quantum head
        self.head_proj = nn.Linear(embed_dim, 1)
        # Quantum expectation head
        backend = Aer.get_backend("aer_simulator")
        self.quantum_circuit = QuantumCircuit(n_qubits_head, backend, shots)
        self.shift = np.pi / 2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        if self.use_transformer:
            seq = x.unsqueeze(1)  # [B, 1, E]
            # Apply quantum transformer per token
            transformed = []
            for token in seq.unbind(dim=1):
                qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
                transformed.append(self.q_transformer(token, qdev))
            seq = torch.stack(transformed, dim=1)
            x = seq.squeeze(1)
        x = self.head_proj(x)  # [B, 1]
        x = x.squeeze(-1)      # [B]
        logits = HybridFunction.apply(x, self.quantum_circuit, self.shift)
        probs = torch.sigmoid(logits).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridNet"]
