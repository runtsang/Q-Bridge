import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of quantum states of the form
    cos(θ)|0...0⟩ + e^{iφ} sin(θ)|1...1⟩.  The labels are a richer
    non‑linear function of (θ,φ) so that the quantum model has to learn
    higher‑order correlations.
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
    labels = np.sin(2 * thetas) * np.cos(phis) + 0.2 * np.sin(4 * thetas)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Torch dataset that yields a dictionary with quantum state tensors
    and their corresponding regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression(tq.QuantumModule):
    """
    Quantum regression model that combines a trainable feature‑encoding
    layer, several layers of parameterised rotations and entanglement,
    and a classical head.  The circuit depth and entangling pattern are
    configurable to explore expressivity‑vs‑noise trade‑offs.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 2):
            super().__init__()
            self.num_wires = num_wires
            self.depth = depth
            self.layers = nn.ModuleList()
            for _ in range(depth):
                # Random rotation layer
                self.layers.append(tq.RandomLayer(n_ops=20, wires=list(range(num_wires))))
                # Entangling block (CX ladder)
                self.layers.append(tq.CX(wires=[(i, i + 1) for i in range(num_wires - 1)]))
                # Parameterised rotations on each wire
                self.layers.append(tq.RX(has_params=True, trainable=True))
                self.layers.append(tq.RY(has_params=True, trainable=True))

        def forward(self, qdev: tq.QuantumDevice) -> None:
            for layer in self.layers:
                layer(qdev)

    def __init__(
        self,
        num_wires: int,
        depth: int = 2,
        measurement: str = "PauliZ",
        use_advanced_encoder: bool = True,
    ):
        super().__init__()
        self.num_wires = num_wires
        # Encoder: if advanced, use a Ry‑only encoder; otherwise a generic RyRx
        encoder_name = f"{num_wires}xRy" if use_advanced_encoder else f"{num_wires}xRyRx"
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.q_layer = self.QLayer(num_wires, depth)
        # Measurement: PauliZ or both Z and X
        if measurement == "PauliZ":
            self.measure = tq.MeasureAll(tq.PauliZ)
            feature_dim = num_wires
        else:
            self.measure = tq.MeasureAll([tq.PauliZ, tq.PauliX])
            feature_dim = num_wires * 2
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Run the quantum circuit on a batch of input states and return
        the regression prediction.  The function is fully differentiable
        with respect to all trainable parameters.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data into quantum amplitudes
        self.encoder(qdev, state_batch)
        # Apply trainable circuit
        self.q_layer(qdev)
        # Extract expectation values
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
