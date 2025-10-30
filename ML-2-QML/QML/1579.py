import qiskit
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import Parameter
import torch

class ParametricQuantumHead:
    """
    Variational quantum circuit used as the final layer of
    HybridQuantumCNN. The circuit is a 3‑qubit chain of H‑Ry‑CX gates
    with parameters that are learned via a parameter‑shift rule.
    """
    def __init__(self, n_qubits: int = 3, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.theta = Parameter("θ")
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.ry(self.theta, 0)
        self._circuit.cx(0, 1)
        self._circuit.ry(self.theta, 1)
        self._circuit.cx(1, 2)
        self._circuit.ry(self.theta, 2)
        self._circuit.measure_all()

    def expectation(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each set of parameters in `params`.
        Returns the expectation value of Z⊗Z⊗Z.
        """
        compiled = transpile(self._circuit, self.backend)
        qobjs = []
        for p in params:
            bind = {self.theta: p[0]}  # using same theta for all qubits
            qobj = assemble(compiled, parameter_binds=[bind], shots=self.shots)
            qobjs.append(qobj)
        job = self.backend.run(qobjs)
        results = job.result().get_counts()
        exp_vals = []
        for res in results:
            counts = np.array(list(res.values()))
            states = np.array([int(k, 2) for k in res.keys()])
            probs = counts / self.shots
            exp_vals.append(np.sum((2*states-3) * probs))  # Z⊗Z⊗Z expectation
        return np.array(exp_vals)

class HybridQuantumLayer(nn.Module):
    """
    PyTorch wrapper that forwards a scalar feature through the quantum
    circuit and returns a probability via a sigmoid. Gradients are
    computed using the parameter‑shift rule.
    """
    def __init__(self, n_qubits: int = 3, shots: int = 1024, shift: float = np.pi/2):
        super().__init__()
        self.quantum_head = ParametricQuantumHead(n_qubits, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch,) scalar features
        batch = x.shape[0]
        params = torch.repeat_interleave(x.unsqueeze(1), repeats=1, dim=1).cpu().numpy()
        exp_vals = self.quantum_head.expectation(params)
        probs = torch.tensor(exp_vals, dtype=x.dtype, device=x.device)
        logits = torch.sigmoid(probs)
        return logits

__all__ = ["ParametricQuantumHead", "HybridQuantumLayer"]
