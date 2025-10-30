import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

def generate_superposition_data(num_wires: int, samples: int, seed: int | None = None, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset of amplitude‑encoded states with optional Gaussian noise.
    Parameters
    ----------
    num_wires : int
        Number of qubits / wires.
    samples : int
        Number of samples.
    seed : int | None
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of Gaussian noise added to the labels.
    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_wires)
        Complex amplitude vectors.
    labels : np.ndarray of shape (samples,)
        Regression targets.
    """
    rng = np.random.default_rng(seed)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega0 = np.zeros(2 ** num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2 ** num_wires, dtype=complex)
        omega1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_std > 0:
        labels += rng.normal(scale=noise_std, size=labels.shape)
    return states.astype(complex), labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset returning amplitude‑encoded quantum states and regression targets.
    """
    def __init__(self, samples: int, num_wires: int, seed: int | None = None, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed, noise_std)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.
    A variational circuit with trainable rotation layers and configurable entanglement
    is wrapped into a TorchLayer for seamless PyTorch integration.
    """
    def __init__(self,
                 num_wires: int,
                 num_layers: int = 3,
                 entanglement: str = "cnot",
                 backend: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.entanglement = entanglement
        self.dev = qml.device(backend, wires=num_wires, shots=None)

        weight_shapes = {"weights": (num_layers, num_wires, 3)}

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(state_vector, weights):
            qml.QubitStateVector(state_vector, wires=range(num_wires))
            for layer_idx in range(num_layers):
                for wire in range(num_wires):
                    w = weights[layer_idx, wire]
                    qml.RX(w[0], wires=wire)
                    qml.RY(w[1], wires=wire)
                    qml.RZ(w[2], wires=wire)
                # Entanglement pattern
                if entanglement == "cnot":
                    for wire in range(num_wires - 1):
                        qml.CNOT(wires=[wire, wire + 1])
                elif entanglement == "full":
                    for i in range(num_wires):
                        for j in range(i + 1, num_wires):
                            qml.CNOT(wires=[i, j])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_wires)]

        self.qnn = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        features = self.qnn(state_batch)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
