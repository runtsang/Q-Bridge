import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """Quantum‑inspired hybrid network that uses a 2×2 quantum filter
    followed by a variational classifier.  The filter is a small
    random circuit that encodes the image patch into qubit angles
    and measures the average |1> probability.  The classifier
    implements a layered ansatz with explicit encoding and
    CZ‑entanglement, mirroring the classical architecture."""

    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 3,
        num_qubits: int = 4,
        num_classes: int = 10,
        shots: int = 1024,
        backend=None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.depth = depth
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.shots = shots

        # Quantum filter circuit
        self.filter_circuit = self._build_filter_circuit()

        # Variational classifier circuit
        self.classifier_circuit, self.encoding, self.weights, self.observables = self._build_classifier_circuit()

        # Backend
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def _build_filter_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        theta = ParameterVector("theta", self.num_qubits)
        for i in range(self.num_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(self.num_qubits, 2)
        qc.measure_all()
        return qc

    def _build_classifier_circuit(self):
        n = self.num_qubits
        encoding = ParameterVector("x", n)
        weights = ParameterVector("w", n * self.depth)

        qc = QuantumCircuit(n)
        for i, p in enumerate(encoding):
            qc.rx(p, i)

        w_idx = 0
        for _ in range(self.depth):
            for q in range(n):
                qc.ry(weights[w_idx], q)
                w_idx += 1
            for q in range(n - 1):
                qc.cz(q, q + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (n - i - 1)) for i in range(n)
        ]
        return qc, encoding, weights, observables

    def _quantum_filter(self, patch: np.ndarray) -> float:
        """Evaluate the filter circuit on a single 2×2 patch."""
        data = patch.flatten()
        param_binds = {f"theta{i}": np.pi if val > 0.5 else 0 for i, val in enumerate(data)}
        job = qiskit.execute(
            self.filter_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result()
        counts = result.get_counts(self.filter_circuit)
        prob = sum(int(bit) * c for bit, c in counts.items())
        return prob / (self.shots * self.num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (B, 1, H, W)
        bsz, _, h, w = x.shape
        patch_probs = []
        for i in range(0, h, self.kernel_size):
            for j in range(0, w, self.kernel_size):
                patch = x[:, 0, i : i + self.kernel_size, j : j + self.kernel_size]
                patch_np = patch.detach().cpu().numpy()
                probs = np.array([self._quantum_filter(p) for p in patch_np])
                patch_probs.append(probs)
        features = torch.tensor(patch_probs, dtype=torch.float32, device=x.device).view(bsz, -1)

        # Classical surrogate for the variational classifier
        # In a full quantum implementation this would bind the
        # features to the encoding parameters and measure observables.
        linear = nn.Linear(features.size(1), self.num_classes, bias=False).to(x.device)
        logits = linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
