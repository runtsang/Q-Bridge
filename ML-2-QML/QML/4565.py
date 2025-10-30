import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

class QuantumCircuitWrapper:
    """
    Builds a layered ansatz with explicit encoding and variational parameters.
    Provides a run method that returns the expectation of Z on the first qubit.
    """
    def __init__(self, num_qubits: int, depth: int, backend=None, shots: int = 512):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self._compiled = transpile(self.circuit, self.backend)

    def _build_circuit(self):
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circ = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circ.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circ.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circ.cz(qubit, qubit + 1)
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return circ, list(encoding), list(weights), observables

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for the given variational parameters and return
        the expectation value of the first Z observable.
        """
        bind_dict = {self.weights[i]: params[i] for i in range(len(params))}
        qobj = assemble(self._compiled, parameter_binds=[bind_dict], shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        state = result.get_statevector(self.circuit)
        # Build full Z operator for the first qubit
        z = np.array([[1, 0], [0, -1]])
        z_full = z
        for _ in range(self.num_qubits - 1):
            z_full = np.kron(z_full, np.eye(2))
        expectation = np.real(np.dot(state.conj(), z_full @ state))
        return np.array([expectation])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards the classical tensor through the quantum
    circuit and implements a parameter‑shift gradient approximation.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        np_inputs = inputs.detach().cpu().numpy()
        exp_list = []
        for vec in np_inputs:
            exp = ctx.circuit.run(vec + shift)
            exp_list.append(exp)
        out = torch.tensor(np.concatenate(exp_list), dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        np_inputs = inputs.detach().cpu().numpy()
        grad_list = []
        for vec in np_inputs:
            grad_vec = []
            for i in range(len(vec)):
                plus = np.copy(vec)
                minus = np.copy(vec)
                plus[i] += shift
                minus[i] -= shift
                exp_plus = ctx.circuit.run(plus)
                exp_minus = ctx.circuit.run(minus)
                grad_vec.append(exp_plus - exp_minus)
            grad_list.append(grad_vec)
        grads = torch.tensor(np.array(grad_list), dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class HybridQuantumLayer(nn.Module):
    """
    PyTorch module that applies a quantum expectation value to a tensor.
    """
    def __init__(self, num_qubits: int, depth: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(num_qubits, depth)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class SamplerQNN:
    """
    Simple sampler that returns statevector probabilities for a given parameter set.
    """
    def __init__(self, circuit: QuantumCircuit, backend=None, shots: int = 1024):
        self.circuit = circuit
        self.backend = backend or AerSimulator()
        self.shots = shots

    def sample(self, params: np.ndarray) -> np.ndarray:
        bind_dict = {self.circuit.parameters[i]: params[i] for i in range(len(params))}
        qobj = assemble(self.circuit, parameter_binds=[bind_dict], shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        probs = np.zeros(2 ** self.circuit.num_qubits)
        for state, cnt in counts.items():
            probs[int(state, 2)] = cnt / self.shots
        return probs

__all__ = ["QuantumCircuitWrapper", "HybridQuantumLayer", "HybridFunction", "SamplerQNN"]
