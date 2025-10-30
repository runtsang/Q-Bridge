import torch
import pennylane as qml
from pennylane import numpy as np

# Two‑qubit device for the variational circuit
dev = qml.device("default.qubit", wires=2)

def create_sampler_qnn():
    """
    Returns a hybrid QNN that samples from a 2‑qubit variational circuit.
    The ansatz uses Ry rotations on each qubit followed by a CNOT chain,
    then another layer of Ry rotations.  
    Input parameters are two angles, and the circuit has four trainable
    weight parameters.
    """
    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray, weights: np.ndarray):
        # Input rotations
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)

        # Entangling layer
        qml.CNOT(wires=[0, 1])

        # Weight rotations
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)

        # Second entangling layer
        qml.CNOT(wires=[0, 1])

        # Final weight rotations
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)

        # Return expectation values of Pauli‑Z
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    class SamplerQNN:
        """
        Wrapper exposing a torch‑compatible interface for the QNN.
        """
        def __init__(self):
            # Initialise trainable weight parameters
            self.weights = torch.nn.Parameter(torch.randn(4))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Execute the circuit and convert expectation values
            to probability outputs in [0,1].
            """
            probs = circuit(inputs.detach().numpy(), self.weights.detach().numpy())
            probs = torch.tensor(probs, dtype=torch.float32)
            # Convert expectation values to probabilities
            probs = (probs + 1) / 2
            return probs

    return SamplerQNN()

__all__ = ["create_sampler_qnn"]
