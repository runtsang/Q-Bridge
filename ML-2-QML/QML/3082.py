import pennylane as qml
import torch
import numpy as np
from typing import Tuple

class QuantumFCLCircuit:
    """
    Parameterised quantum circuit that implements a single‑qubit
    fully‑connected layer.  The circuit applies a trainable rotation
    followed by an optional entangling layer and measures the Pauli‑Z
    expectation value.
    """
    def __init__(self, num_qubits: int = 1, dev_name: str = "default.qubit"):
        self.dev = qml.device(dev_name, wires=num_qubits)
        @qml.qnode(self.dev, interface="torch")
        def circuit(params, x):
            # Encode classical input into rotation angles
            qml.RX(x[0], wires=0)
            # Variational block
            for i in range(num_qubits):
                qml.RY(params[i], wires=i)
            # Simple entanglement
            for i in range(num_qubits-1):
                qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def __call__(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(params, x)

class QuantumGateQNode:
    """
    Single‑qubit gate implemented as a PennyLane QNode.
    The gate takes a scalar input and returns a scalar output,
    mimicking the behaviour of a quantum‑gate‑based LSTM gate.
    """
    def __init__(self, dev_name: str = "default.qubit"):
        self.dev = qml.device(dev_name, wires=1)
        @qml.qnode(self.dev, interface="torch")
        def gate(params, x):
            qml.RX(x[0], wires=0)
            qml.RY(params[0], wires=0)
            return qml.expval(qml.PauliZ(0))
        self.gate = gate

    def __call__(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.gate(params, x)

class QuantumLSTMCircuit:
    """
    Variational quantum circuit that implements the four gates of an LSTM.
    Each gate is represented by a `QuantumGateQNode` and the circuit
    accepts an input tensor containing the concatenated previous hidden
    state and current input.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_gate = QuantumGateQNode()
        self.input_gate  = QuantumGateQNode()
        self.update_gate = QuantumGateQNode()
        self.output_gate = QuantumGateQNode()

    def __call__(self, params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = torch.sigmoid(self.forget_gate(params[0], x))
        i = torch.sigmoid(self.input_gate(params[1], x))
        g = torch.tanh(self.update_gate(params[2], x))
        o = torch.sigmoid(self.output_gate(params[3], x))
        return f, i, g, o

__all__ = ["QuantumFCLCircuit", "QuantumGateQNode", "QuantumLSTMCircuit"]
