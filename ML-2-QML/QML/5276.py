"""
Quantum regression model inspired by EstimatorQNN and QuantumRegression.
The circuit encodes the input state, applies a random layer and trainable rotations, then
measures Pauli‑Z on each qubit and maps the result through a classical linear head.
"""

import torch
import torch.nn as nn
import torchquantum as tq
from.QuantumRegression import generate_superposition_data

class EstimatorQNNGen160(tq.QuantumModule):
    """
    Quantum neural network that processes 2^n‑dimensional input states.
    Supports 160 training states (support vectors) for data‑driven initialization.
    """
    class QLayer(tq.QuantumModule):
        """Variational layer with random gates followed by trainable rotations."""
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 4, num_support: int = 160) -> None:
        super().__init__()
        self.n_wires = num_wires

        # Encoder that maps a classical state vector into |0...0> + |1...1> superposition
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Variational layer
        self.q_layer = self.QLayer(num_wires)

        # Measurement of all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear head mapping qubit expectation values to scalar output
        self.head = nn.Linear(num_wires, 1)

        # Support vectors and labels for potential data‑driven initialization
        states, labels = generate_superposition_data(num_wires, num_support)
        self.register_buffer("support_states", torch.tensor(states, dtype=torch.cfloat))
        self.register_buffer("support_labels", torch.tensor(labels, dtype=torch.float32))

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_batch: Tensor of shape (batch, 2**n_wires) with complex entries.
        Returns:
            Tensor of shape (batch,) with regression predictions.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode classical state into the quantum device
        self.encoder(qdev, state_batch)

        # Variational processing
        self.q_layer(qdev)

        # Measurement
        features = self.measure(qdev)  # (batch, n_wires)

        # Classical linear head
        return self.head(features).squeeze(-1)

__all__ = ["EstimatorQNNGen160"]
