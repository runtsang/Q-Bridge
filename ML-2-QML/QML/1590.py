import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.
    Adds Gaussian noise to the labels to mirror the classical noise model.
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
    labels += np.random.normal(0.0, noise_std, size=labels.shape)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset yielding quantum state tensors and regression targets.
    """
    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.05):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise_std)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(tq.QuantumModule):
    """
    Variational circuit with entangling layers and a classical linear head.
    Supports configurable depth and number of layers.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, n_layers: int = 2):
            super().__init__()
            self.n_wires = num_wires
            self.n_layers = n_layers
            self.base_layer = tq.RandomLayer(n_ops=15, wires=list(range(num_wires)))
            self.rotation = tq.RX(has_params=True, trainable=True)
            self.entangle = tq.CNOT(has_params=False)

        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                self.base_layer(qdev)
                for w in range(self.n_wires):
                    self.rotation(qdev, wires=w)
                # entangle adjacent qubits
                for w in range(self.n_wires - 1):
                    self.entangle(qdev, control_wire=w, target_wire=w + 1)

    def __init__(self, num_wires: int, n_layers: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, n_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
