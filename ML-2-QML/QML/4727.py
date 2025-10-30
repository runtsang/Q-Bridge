import numpy as np
import qiskit
from qiskit import assemble, transpile
import torch
import torch.nn as nn

class QuantumCircuit:
    """
    Parameterised two‑qubit circuit that matches the structure used in the
    original FCL example but wrapped for batch execution.
    """
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
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
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit.
    Finite‑difference gradients are used to keep the implementation
    simple while still supporting back‑prop through the circuit.
    """
    @staticmethod
    def forward(ctx, angles: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(angles)
        angles_np = angles.detach().cpu().numpy()
        expectation = circuit.run(angles_np)
        return torch.tensor(expectation, device=angles.device, dtype=angles.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        angles, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for a in angles:
            a_np = a.item()
            exp_right = circuit.run([a_np + shift])
            exp_left  = circuit.run([a_np - shift])
            grads.append((exp_right - exp_left) / (2 * shift))
        grad_angles = torch.tensor(grads, device=angles.device, dtype=angles.dtype)
        return grad_angles * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards input angles through the quantum circuit.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, qiskit.Aer.get_backend("qasm_simulator"), shots)
        self.shift = shift

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(angles, self.circuit, self.shift)

class HybridFullyConnectedLayerQML(nn.Module):
    """
    Quantum‑backed counterpart of the classical HybridFullyConnectedLayer.
    It accepts a tensor of angles produced by the classical front‑end
    and yields a probability vector that can be concatenated with its
    complement for binary classification.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.hybrid = Hybrid(n_qubits, shots, shift)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        probs = self.hybrid(angles).squeeze()
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "HybridFullyConnectedLayerQML"]
