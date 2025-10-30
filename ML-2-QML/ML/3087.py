import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Optional

class UnifiedQuantumModel(nn.Module):
    """
    Hybrid classifier that combines a classical feed‑forward backbone with a
    quantum variational circuit.  The classical network extracts features,
    whereas the quantum circuit operates on those features and produces a
    two‑class score that is fused with a classical head.  The depth of both
    parts is matched to enable a direct comparison and joint training.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        n_qubits: Optional[int] = None,
        quantum_backend: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of input features (or qubits if a quantum backend is used).
        depth : int
            Depth of the classical and quantum layers.
        n_qubits : int | None
            Number of qubits for the quantum circuit.  If ``None`` the
            model runs purely classically.
        quantum_backend : str | None
            Identifier of a Qiskit or Pennylane backend to run the quantum
            circuit.  If omitted, the quantum part is disabled.
        """
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.n_qubits = n_qubits or num_features
        self.quantum_backend = quantum_backend

        # Classical backbone: depth‑scaled fully‑connected network
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(num_features, 2))
        self.classical_net = nn.Sequential(*layers)

        # Quantum parameters (only used if a backend is supplied)
        if self.quantum_backend:
            # Variational parameters for the ansatz
            self.weights = nn.Parameter(
                torch.randn(self.n_qubits * depth)
            )
            self.encoding = nn.Parameter(
                torch.randn(self.n_qubits)
            )
            # Linear head that maps the quantum expectation vector to logits
            self.quantum_head = nn.Linear(self.n_qubits, 2)
            # Optional quantum simulator
            try:
                from qiskit import Aer
                self.sim = Aer.get_backend("qasm_simulator")
            except Exception as exc:
                raise RuntimeError(
                    f"Quantum backend '{self.quantum_backend}' not available: {exc}"
                ) from exc
        else:
            self.weights = None
            self.encoding = None
            self.quantum_head = None
            self.sim = None

    # ------------------------------------------------------------------
    # Helper: run a single‑shot quantum circuit and return expectation
    # values of Pauli‑Z on each qubit.
    # ------------------------------------------------------------------
    def _quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a simple variational circuit on the batch ``x``.  The circuit
        is constructed once and reused.  For each qubit we apply an RX
        encoding followed by ``depth`` layers of RY rotations and CZ
        entanglement.  The expectation of Pauli‑Z on each qubit is returned.
        """
        if not hasattr(self, "_qc"):
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(self.n_qubits)
            # RX encoding
            for idx, param in enumerate(self.encoding):
                qc.rx(param, idx)
            # Depth‑scaled ansatz
            for d in range(self.depth):
                for w in range(self.n_qubits):
                    qc.ry(self.weights[d * self.n_qubits + w], w)
                for w in range(self.n_qubits - 1):
                    qc.cz(w, w + 1)
            self._qc = qc

        # Run the circuit on the simulator
        from qiskit import execute
        job = execute(self._qc, self.sim, shots=1)
        result = job.result()
        counts = result.get_counts()
        exp_vals: List[float] = []
        total_shots = sum(counts.values())
        for qubit in range(self.n_qubits):
            exp = 0.0
            for bitstring, freq in counts.items():
                # Pauli‑Z: +1 for |0>, -1 for |1>
                bit = int(bitstring[self.n_qubits - 1 - qubit])
                exp += freq * (1.0 if bit == 0 else -1.0)
            exp_vals.append(exp / total_shots)
        return torch.tensor(exp_vals, device=x.device, dtype=torch.float)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that combines the classical logits with the quantum
        logits (if a backend is provided).  The two logits are summed
        element‑wise to produce the final output.
        """
        cls_logits = self.classical_net(x)
        if self.quantum_backend:
            q_exp = self._quantum_forward(x)  # shape: (n_qubits,)
            q_logits = self.quantum_head(q_exp)  # shape: (2,)
            return cls_logits + q_logits
        return cls_logits

__all__ = ["UnifiedQuantumModel"]
