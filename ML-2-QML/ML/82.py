import numpy as np
import torch
from torch import nn
import pennylane as qml
import pennylane.numpy as pnp

class ConvFilterHybrid(nn.Module):
    """Hybrid convolutional filter supporting classical, quantum, or hybrid execution.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    mode : {'c', 'q', 'q2c'}, default 'c'
        Execution mode: 'c' for classical convolution, 'q' for quantum-only,
        'q2c' for hybrid where quantum output is added as a bias.
    threshold : float, default 0.0
        Threshold for binarizing input data before quantum encoding.
    weight_init : bool, default True
        Whether to initialize a learnable weight matrix for the quantum part.
    """
    def __init__(self, kernel_size: int = 2, mode: str = "c",
                 threshold: float = 0.0, weight_init: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode
        self.threshold = threshold

        # Classical convolution
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True) if mode!= "q" else None

        # Quantum circuit
        self.n_qubits = kernel_size ** 2
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Trainable parameters for variational ansatz
        self.ansatz_params = nn.Parameter(torch.randn(self.n_qubits))

        # Optional linear weight to combine conv and quantum outputs
        if weight_init:
            self.fc = nn.Linear(1, 1, bias=False)
        else:
            self.fc = None

    def _quantum_layer(self, data: np.ndarray) -> float:
        """Run the variational circuit on the input data."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Encode data into rotations
            for i, val in enumerate(x):
                qml.RX(np.pi if val > self.threshold else 0.0, wires=i)
            # Variational ansatz
            for i, p in enumerate(params):
                qml.RZ(p, wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of Z on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        x_flat = torch.tensor(data.reshape(-1), dtype=torch.float32)
        outputs = circuit(x_flat, self.ansatz_params)
        # Convert expectation values to probabilities of |1>
        probs = (1 - torch.stack(outputs)) / 2
        return probs.mean().item()

    def forward(self, data: np.ndarray) -> float:
        """Forward pass for the selected mode."""
        # Classical path
        if self.mode!= "q":
            tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            classical_output = activations.mean().item()
        else:
            classical_output = 0.0

        # Quantum path
        quantum_output = self._quantum_layer(data)

        # Hybrid combination
        if self.mode == "q2c":
            combined = classical_output + quantum_output
            if self.fc is not None:
                combined = self.fc(torch.tensor([combined]))
            return combined.item()
        # Return according to mode
        return quantum_output if self.mode == "q" else classical_output
