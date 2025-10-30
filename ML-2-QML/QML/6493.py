"""ConvGen: Parameter‑efficient variational quanvolution.

This quantum module mirrors the classical ConvGen but uses a
parameter‑efficient PennyLane variational circuit.  The circuit
consists of data‑dependent RY rotations followed by a
parameterised entangling layer.  The output is the average
probability of measuring |1> across all qubits.  A simple
gradient‑based training loop is provided for quick experiments.
"""

import pennylane as qml
import numpy as np
import torch
from torch import nn


class ConvGen(nn.Module):
    """Quantum variational convolution implemented with PennyLane."""

    def __init__(self,
                 kernel_size: int = 2,
                 shots: int = 1000,
                 device_name: str = "default.qubit.autograd"):
        """
        Args:
            kernel_size: Size of the quantum filter grid.
            shots: Number of shots for measurement.
            device_name: PennyLane device name.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.dev = qml.device(device_name, wires=self.n_qubits, shots=shots)
        # Parameter vector for the variational layer
        self.params = nn.Parameter(torch.randn(self.n_qubits))
        self.loss_fn = nn.MSELoss()
        # Create QNode once
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

    def circuit(self, data, params):
        """Variational circuit with data‑dependent RY and parameterised RZ."""
        for i in range(self.n_qubits):
            qml.RY(data[i] * np.pi, wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(self.n_qubits):
            qml.RZ(params[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: evaluate the circuit on batch of data and return
        the mean probability of measuring |1> across qubits.
        """
        # Convert input to numpy array
        data = x.detach().cpu().numpy().reshape(-1)
        # Evaluate circuit
        probs = self.qnode(data, self.params)
        # Convert from expectation of PauliZ to probability of |1>
        probs = (1 - probs) / 2
        return probs.mean()

    def run(self, data) -> float:
        """
        Run the quantum filter on a 2D array of shape
        (kernel_size, kernel_size) and return the scalar output.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).view(
            1, self.n_qubits)
        return self.forward(tensor).item()

    def fit(self,
            data: torch.Tensor,
            targets: torch.Tensor,
            lr: float = 1e-3,
            epochs: int = 100,
            callback: callable | None = None) -> None:
        """
        Gradient‑based training loop that optimises the variational
        parameters to minimise the hybrid loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            if callback:
                callback(epoch, loss.item(), outputs.detach().cpu().numpy())
