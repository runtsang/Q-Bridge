import numpy as np
import qiskit
import torch
import torch.nn as nn
from qiskit import execute, assemble, transpile

class QuantumPatchFilter:
    """
    Quantum filter that operates on 2×2 image patches.  It implements the
    filter from the original QML Conv example but exposes a batch‑level
    interface that accepts a torch.Tensor of shape (B, C, H, W).  Each
    patch is converted to a set of rotation angles, bound to the circuit
    parameters and executed on a QASM simulator.  The returned value for
    a patch is the average probability of measuring |1> across all qubits.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 100):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def _run_single(self, data: np.ndarray) -> float:
        bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(data)}
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts()
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    def run_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: torch.Tensor of shape (B, C, H, W) with values in [0,255].
        Returns a tensor of shape (B, num_patches) where each entry is the
        quantum filter output for one patch.
        """
        B, C, H, W = images.shape
        patch_size = self.kernel_size
        patches_per_row = H // patch_size
        patches_per_col = W // patch_size
        num_patches = patches_per_row * patches_per_col
        outputs = torch.empty((B, num_patches), dtype=torch.float32)
        for b in range(B):
            patch_idx = 0
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    patch = images[b, 0, i:i+patch_size, j:j+patch_size]
                    data = patch.view(-1).cpu().numpy()
                    outputs[b, patch_idx] = self._run_single(data)
                    patch_idx += 1
        return outputs

class QuantumCircuit:
    """Parametrised two‑qubit circuit used by the hybrid head."""
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
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
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

class HybridQuantumHead(nn.Module):
    """
    Implements the hybrid quantum expectation head.  It forwards the logits
    through a QuantumCircuit and returns a probability estimate.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100, shift: float = np.pi/2):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, qiskit.Aer.get_backend("aer_simulator"), shots)
        self.shift = shift

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: tensor of shape (B,) or (B,1)
        Returns tensor of shape (B,) with probabilities in [0,1].
        """
        thetas = (logits + self.shift).detach().cpu().numpy()
        expectations = self.quantum_circuit.run(thetas)
        return torch.tensor(expectations, device=logits.device, dtype=torch.float32)
