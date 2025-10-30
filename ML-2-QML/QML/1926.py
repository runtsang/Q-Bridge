"""HybridQFCModel: a quantum variation of the classical CNN with a variational quantum circuit."""
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQFCModel(nn.Module):
    """Quantum hybrid model with a classical encoder and a variational quantum circuit."""
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, shots: int = 1024, use_noise: bool = False, device_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        # Quantum device
        if use_noise:
            # Optional noisy backend (requires IBMQ account)
            self.dev = qml.device("qiskit.ibmq.simulator", backend="qasm_simulator", shots=shots, wires=n_qubits)
        else:
            self.dev = qml.device(device_name, wires=n_qubits, shots=shots)
        # Trainable parameters for the variational circuit
        self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits, 2))
        # Define QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x):
            # Encode classical data into qubits via RX rotations
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RZ(self.q_params[l, i, 0], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                for i in range(n_qubits):
                    qml.RY(self.q_params[l, i, 1], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.circuit = circuit
        self.norm = nn.BatchNorm1d(n_qubits)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)  # shape (batch, 16)
        encoded = pooled[:, :self.n_qubits]  # use first n_qubits features
        # Run circuit for each sample in the batch
        out = torch.stack([self.circuit(encoded[i]) for i in range(bsz)], dim=0)
        return self.norm(out)

__all__ = ["HybridQFCModel"]
