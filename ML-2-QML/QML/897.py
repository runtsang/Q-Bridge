import pennylane as qml
import torch

class QCNet(torch.nn.Module):
    """Pure quantum layer that maps a feature vector to a probability."""
    def __init__(self, num_qubits: int = 4, device: str = "default.qubit"):
        super().__init__()
        self.num_qubits = num_qubits
        self.device = device
        self.dev = qml.device(device, wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter_shift")
        def circuit(theta: torch.Tensor) -> torch.Tensor:
            qml.Hadamard(wires=range(num_qubits))
            for i, wire in enumerate(range(num_qubits)):
                qml.RY(theta[i], wires=wire)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim()!= 2:
            raise ValueError("Input must be 2‑D: (batch, features)")
        params = x[:, :self.num_qubits]
        out = self.circuit(params)
        return torch.sigmoid(out).unsqueeze(-1)
