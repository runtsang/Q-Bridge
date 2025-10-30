import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create quantum states |ψ(θ,ϕ)⟩ = cos θ |0…0⟩ + e^{iϕ} sin θ |1…1⟩
    together with a smooth regression target sin(2θ)·cosϕ.
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

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset returning quantum state tensors and regression targets.
    ``mode`` determines whether the state is returned as a complex vector
    (quantum) or as a real amplitude magnitude vector (classical).
    """
    def __init__(self, samples: int, num_wires: int, mode: str = "quantum"):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
        self.mode = mode

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        if self.mode == "quantum":
            return {
                "states": torch.tensor(self.states[index], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[index], dtype=torch.float32),
            }
        else:  # classical mode
            state = torch.abs(torch.tensor(self.states[index], dtype=torch.cfloat))
            return {
                "features": state,
                "target": torch.tensor(self.labels[index], dtype=torch.float32),
            }

class QuantumFullyConnectedLayer(tq.QuantumModule):
    """
    Parameterised single‑qubit circuit that mimics the classical fully
    connected layer.  It applies an Ry gate with a parameter that is
    fed by the input tensor.
    """
    def __init__(self, n_qubits: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = tq.Circuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(tq.Parameter(), range(n_qubits))
        self.circuit.measure_all()

    def forward(self, qdev: tq.QuantumDevice):
        # Apply the circuit; the parameter will be bound in the caller
        self.circuit(qdev)
        return qdev.expectation(tq.PauliZ)

class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum regression model that seamlessly integrates a feature encoder,
    a variational quantum layer, optional fully connected circuit, and a
    classical read‑out head.  The architecture mirrors the classical
    counterpart but leverages quantum state preparation and measurement.
    """
    def __init__(self, num_wires: int, hidden: int = 32, use_fcl: bool = False):
        super().__init__()
        self.num_wires = num_wires
        self.use_fcl = use_fcl
        # Feature encoder: map each input qubit to an Ry rotation
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = tq.QuantumModule()
        self.q_layer.register_module("random_layer", tq.RandomLayer(n_ops=30, wires=list(range(num_wires))))
        self.q_layer.register_module("rx", tq.RX(has_params=True, trainable=True))
        self.q_layer.register_module("ry", tq.RY(has_params=True, trainable=True))
        if use_fcl:
            self.q_layer.register_module("fcl", QuantumFullyConnectedLayer(num_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # Encode the input state
        self.encoder(qdev, state_batch)
        # Apply the variational layer
        self.q_layer.random_layer(qdev)
        for wire in range(self.num_wires):
            self.q_layer.rx(qdev, wires=wire)
            self.q_layer.ry(qdev, wires=wire)
        if self.use_fcl:
            self.q_layer.fcl(qdev)
        # Measurement
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# expose original names for compatibility
QModel = HybridRegressionModel
RegressionDataset = RegressionDataset
generate_superposition_data = generate_superposition_data

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "QuantumFullyConnectedLayer"]
