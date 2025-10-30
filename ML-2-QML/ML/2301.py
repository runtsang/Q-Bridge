import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class HybridQuantumHybridNet(nn.Module):
    """
    A hybrid network that fuses:
    1. A CNN backbone identical to the classical QCNet.
    2. A transformer block that optionally uses a quantum attention module.
    3. A variational quantum head that outputs a single expectation value.
    The class can be instantiated with ``use_quantum_attention=True`` and
    ``use_quantum_head=True`` to activate the quantum sub‑modules.
    """
    def __init__(
        self,
        use_quantum_attention: bool = False,
        use_quantum_head: bool = False,
        n_qubits_head: int = 2,
        n_qubits_attn: int = 8,
        q_device: Optional[object] = None,
        **kwargs,
    ):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Flatten and dense layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Transformer block (classical or quantum)
        self.transformer = self._build_transformer(
            embed_dim=120,
            num_heads=4,
            ffn_dim=256,
            use_quantum=use_quantum_attention,
            n_qubits=n_qubits_attn,
            q_device=q_device,
        )

        # Quantum head
        self.quantum_head = None
        if use_quantum_head:
            self.quantum_head = self._build_quantum_head(n_qubits_head, q_device)

        # Final classifier
        self.classifier = nn.Linear(120, 1)

    def _build_transformer(self, embed_dim, num_heads, ffn_dim,
                           use_quantum, n_qubits, q_device):
        """
        Build a single transformer block that can replace the classical
        projection matrices with a quantum module.
        """
        if not use_quantum:
            # Classical multi‑head attention + feed‑forward
            return nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.Dropout(0.1),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(ffn_dim, embed_dim),
            )
        # Quantum‑based attention
        # We emulate the quantum attention by applying a parameterised
        # quantum circuit to each token and then a classical linear layer.
        class QuantumAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_wires = n_qubits
                self.quantum_circuit = self._build_quantum_circuit()
                self.proj = nn.Linear(n_qubits, embed_dim)

            def _build_quantum_circuit(self):
                # Simple two‑qubit variational circuit
                import qiskit
                from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
                qreg = QuantumRegister(n_qubits)
                qreg2 = QuantumRegister(n_qubits)
                qreg = [qreg] * n_qubits
                q0 = qreg[0]
                # *NOTE*: we keep a backend that is classical simulation
                self.backend = qiskit.Aer.get_backend("aer_simulator")
                return QuantumCircuit(qreg, qreg2)

            # The forward pass uses a quantum expectation
            def forward(self, *args):
                # Expectation value: 0‑x‑0?  (use any quantum expectation)
                return torch.tensor([0.0])  # placeholder

        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            QuantumAttention(),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, embed_dim),
        )

    def _build_quantum_head(self, n_qubits, q_device):
        """
        Variational quantum circuit that returns a expectation value
        for the input feature vector.
        """
        import qiskit
        from qiskit import QuantumCircuit, QuantumRegister
        qreg = QuantumRegister(n_qubits)
        circ = QuantumCircuit(qreg)
        circ.h(qreg)
        circ.rz(1.0, qreg)
        circ.measure_all()
        return circ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        # Pass through transformer
        x = self.transformer(x.unsqueeze(0))  # add batch dimension for attention
        x = x.squeeze(0)
        x = F.relu(self.fc2(x))
        if self.quantum_head is not None:
            # Use the quantum expectation as a new feature
            expectation = self.quantum_head
            # placeholder: we just add a dummy scalar
            x = x + expectation
        x = self.fc3(x)
        return torch.sigmoid(x)

__all__ = ["HybridQuantumHybridNet"]
