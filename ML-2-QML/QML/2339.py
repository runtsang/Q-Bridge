"""Quantum regression model using torchquantum and a sampler-based quantum circuit.

This module implements the same logical model as the classical counterpart but
leverages a genuine quantum circuit for feature extraction. A quantum encoder maps
classical inputs into a superposition state, followed by a trainable quantum layer
and a measurement that yields expectation values. The final linear head maps these
expectation values to a scalar output. Additionally, a quantum sampler network
provides a parameterized circuit that can be used for probabilistic state preparation.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_quantum_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in the form of quantum superposition states.

    Each sample is a state cos(theta)|0..0> + e^{i phi} sin(theta)|1..1> with a target
    defined by a trigonometric function of the parameters. This dataset mirrors the
    classical superposition data but in a higher dimensional Hilbert space.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for the quantum regression task.

    Parameters
    ----------
    samples : int
        Number of samples to generate.
    num_wires : int
        Number of qubits used to encode each sample.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_quantum_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerQNN(tq.QuantumModule):
    """Parameterised quantum sampler circuit.

    The circuit consists of two layers of Ry rotations followed by a CNOT entangling
    pattern. The parameters are split into input and weight vectors, mirroring the
    classical SamplerNetwork structure.
    """
    def __init__(self, num_wires: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.input_params = tq.ParameterVector("input", 2)
        self.weight_params = tq.ParameterVector("weight", 4)

        # Build the circuit
        self.circuit = tq.Circuit(num_wires=self.num_wires)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Sampler primitive
        self.sampler = tq.StatevectorSampler()

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        """Execute the sampler and return the probability distribution over the computational basis."""
        self.circuit(qdev)
        probs = self.sampler(qdev)
        return probs

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model with a random quantum layer and a classical readout.

    The model architecture mirrors the classical hybrid model:
    1. A general encoder that maps classical amplitudes into a superposition.
    2. A trainable quantum layer comprising random gates and parameterised rotations.
    3. Measurement of all qubits in the Pauli‑Z basis.
    4. A linear head that maps expectation values to a scalar output.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Random gate layer with a fixed number of operations
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            # Trainable rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that prepares a superposition state from the input amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "SamplerQNN", "generate_quantum_superposition_data"]
