import numpy as np
import torch
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit import transpile, assemble

class QuantumAutoencoderCircuit:
    """Variational quantum circuit used as the quantum encoder.
    The circuit consists of a RealAmplitudes ansatz followed by a
    swap‑test style similarity measurement.  The circuit is
    evaluated on a statevector sampler and returns the expectation
    values of Pauli‑Z on every qubit."""
    def __init__(self, num_qubits: int, reps: int = 3, shots: int = 1024):
        self.num_qubits = num_qubits
        self.reps = reps
        self.shots = shots
        self.circuit = RealAmplitudes(num_qubits, reps=reps)
        self.sampler = Sampler()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Return a (batch, num_qubits) array of expectation values."""
        outputs = []
        for sample in inputs:
            circ = self.circuit.copy()
            circ.assign_parameters(dict(zip(circ.parameters, sample)), inplace=True)
            result = self.sampler.run(circ, shots=self.shots).result()
            counts = result.get_counts()
            outputs.append(self._expectation(counts))
        return np.array(outputs)

    def _expectation(self, counts: dict[str, int]) -> np.ndarray:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        exp = []
        for q in range(self.num_qubits):
            mask = 1 << (self.num_qubits - 1 - q)
            prob_ones = np.sum(((states & mask) >> (self.num_qubits - 1 - q)) * probs)
            exp.append(2 * prob_ones - 1)  # map 0->-1, 1->1
        return np.array(exp)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between the classical encoder and the quantum circuit.
    The forward pass evaluates the quantum circuit.  The backward pass
    implements a finite‑difference estimate of the gradient with respect
    to the classical inputs."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumAutoencoderCircuit):
        ctx.circuit = circuit
        outputs = circuit(inputs.detach().cpu().numpy())
        result = torch.tensor(outputs, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        circuit = ctx.circuit
        shift = np.pi / 2
        grad_inputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[1]):
            inp_plus = inputs.clone()
            inp_plus[:, i] += shift
            out_plus = circuit(inp_plus.detach().cpu().numpy())
            inp_minus = inputs.clone()
            inp_minus[:, i] -= shift
            out_minus = circuit(inp_minus.detach().cpu().numpy())
            grad_inputs[:, i] = (out_plus - out_minus) / 2.0
        return grad_output * grad_inputs, None

__all__ = ["QuantumAutoencoderCircuit", "HybridFunction"]
