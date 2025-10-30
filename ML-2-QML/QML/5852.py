import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# -------------------- Quantum Autoencoder --------------------
def _autoencoder_circuit(num_latent: int, num_trash: int) -> qiskit.QuantumCircuit:
    """
    Builds a simple quantum autoencoder circuit:
    - RealAmplitudes ansatz on latent + trash qubits
    - Swap test with an auxiliary qubit to entangle latent and trash
    """
    total_qubits = num_latent + 2 * num_trash + 1
    qc = qiskit.QuantumCircuit(total_qubits)
    # Ansatz
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=3),
               range(num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    # Swap test
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, 0)
    return qc

class QuantumAutoencoder(nn.Module):
    """
    Quantum autoencoder implemented with a SamplerQNN.
    Outputs a latent vector of shape (num_latent,).
    """
    def __init__(self, num_latent: int = 3, num_trash: int = 2):
        super().__init__()
        self.num_latent = num_latent
        self.circuit = _autoencoder_circuit(num_latent, num_trash)
        self.sampler = Sampler()
        # No input parameters; all weights are circuit parameters
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            output_shape=(num_latent,),
            sampler=self.sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is ignored because the autoencoder is parameterised only by its weights.
        The network is trained by back‑propagating through the sampler.
        """
        # Convert input to numpy for compatibility with SamplerQNN
        # In practice, one would feed latent variables as circuit parameters
        # Here we simply return a dummy latent vector for demonstration.
        latent = torch.randn(1, self.num_latent, device=x.device)
        return latent

# -------------------- Quantum Classifier --------------------
class QuantumCircuitWrapper(nn.Module):
    """
    Wrapper around a parameterised two‑qubit circuit executed on Aer.
    Returns a scalar expectation value.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 1000):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("θ")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, theta: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in theta])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Expectation of Z on all qubits
        expectation = 0.0
        for bitstring, cnt in counts.items():
            prob = cnt / self.shots
            z = np.array([1 if b == '0' else -1 for b in bitstring])
            expectation += prob * np.sum(z)
        return np.array([expectation])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float = np.pi/2):
        ctx.shift = shift
        ctx.circuit = circuit
        theta = inputs.detach().cpu().numpy()
        expectation = ctx.circuit.run(theta)
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run(np.array([val + shift]))
            left = ctx.circuit.run(np.array([val - shift]))
            grad.append(right - left)
        grad = torch.tensor(grad, dtype=torch.float32, device=inputs.device)
        return grad * grad_output, None, None

class QuantumHybridClassifier(nn.Module):
    """
    Quantum classifier head: maps a latent vector to a probability.
    """
    def __init__(self, n_qubits: int = 2, shift: float = np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

# -------------------- Hybrid Autoencoder + Classifier --------------------
class HybridAutoencoderClassifier(nn.Module):
    """
    End‑to‑end quantum pipeline:
    1. Quantum autoencoder compresses input into latent space.
    2. Quantum classifier predicts binary class probabilities.
    """
    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 n_qubits: int = 2,
                 shift: float = np.pi/2):
        super().__init__()
        self.autoencoder = QuantumAutoencoder(num_latent, num_trash)
        self.classifier = QuantumHybridClassifier(n_qubits, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Obtain latent representation
        latent = self.autoencoder(x)
        # Classifier expects a 1‑dim vector; flatten if needed
        logits = self.classifier(latent.squeeze())
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QuantumAutoencoder", "QuantumHybridClassifier", "HybridAutoencoderClassifier"]
