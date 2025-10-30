import pennylane as qml
import torch
import torch.nn as nn
import pennylane.numpy as np

class EstimatorQNN(nn.Module):
    """
    Quantum neural network wrapper that extends the original seed.
    - 2 qubits, variational circuit with layer‑wise rotation gates and CNOT entanglement.
    - Input features encoded via angle encoding (RY) on each qubit.
    - Trainable weights are rotation angles for each qubit.
    - Observable: Pauli Z on both qubits (ZZ).
    """
    def __init__(self, num_qubits=2, layers=2, device_name="default.qubit", shots=1024):
        super().__init__()
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device(device_name, wires=num_qubits, shots=shots)

        # Trainable parameters: num_qubits * layers
        self.params = nn.Parameter(torch.randn(num_qubits * layers))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            # Encode inputs
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(layers):
                for i in range(num_qubits):
                    qml.RX(params[idx], wires=i)
                    idx += 1
                # Entanglement pattern (chain)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Wrap-around entanglement
                qml.CNOT(wires=[num_qubits - 1, 0])
            # Measurement
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation value for each input sample.
        inputs: Tensor of shape (batch_size, num_qubits)
        returns: Tensor of shape (batch_size, 1)
        """
        inputs = inputs.float()
        return torch.stack([self.circuit(inputs[i], self.params) for i in range(inputs.shape[0])]).unsqueeze(-1)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(inputs)

__all__ = ["EstimatorQNN"]
