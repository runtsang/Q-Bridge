import torch
from torch import nn
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """
    Quantum LSTM where each gate is realised by a small quantum circuit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self._make_gate()
        self.input_gate = self._make_gate()
        self.update = self._make_gate()
        self.output_gate = self._make_gate()

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _make_gate(self):
        class QGate(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for i, gate in enumerate(self.params):
                    gate(qdev, wires=i)
                for i in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[i, i + 1])
                return self.measure(qdev)
        return QGate(self.n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class HybridAutoencoder(nn.Module):
    """
    Quantum‑classical hybrid autoencoder that mirrors the classical
    architecture but replaces the latent bottleneck with a variational
    quantum circuit and a quantum LSTM for sequence processing.
    """
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 32,
                 n_qubits: int = 5,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits

        algorithm_globals.random_seed = 42
        self.feature_map = ZFeatureMap(input_dim, reps=1)
        self.ansatz = RealAmplitudes(n_qubits, reps=2)

        self.qnn = EstimatorQNN(
            circuit=self.feature_map + self.ansatz,
            observables=SparsePauliOp.from_list([("Z" * n_qubits, 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

        self.lstm = QLSTM(input_dim=latent_dim,
                          hidden_dim=lstm_hidden,
                          n_qubits=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        q_latent = self.qnn(flat)
        q_latent = q_latent.view(x.size(0), 1, -1)
        lstm_out, _ = self.lstm(q_latent)
        recon = self.decoder(lstm_out.squeeze(1))
        return recon

def train_hybrid_quantum_autoencoder(model: nn.Module,
                                     data: torch.Tensor,
                                     *,
                                     epochs: int = 100,
                                     batch_size: int = 64,
                                     lr: float = 1e-3,
                                     device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: torch.Tensor | list[float] | tuple[float,...]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["HybridAutoencoder", "train_hybrid_quantum_autoencoder"]
