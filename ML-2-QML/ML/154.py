import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QiskitEstimator

class EstimatorQNN(nn.Module):
    """
    A hybrid classical‑quantum regressor that trains a small feed‑forward network
    together with a 1‑qubit variational circuit.  The network mirrors the
    original EstimatorQNN but has been extended to support a full training
    pipeline, early‑stopping, and per‑sample loss computation.  The class
    inherits from :class:`torch.nn.Module` so it can be used with any
    PyTorch optimizer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: list[int] = [8, 4],
        output_dim: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        # Classical network
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.classical_net = nn.Sequential(*layers)

        # Quantum circuit
        self.input_param = Parameter("x")
        self.weight_param = Parameter("w")
        self.circuit = QuantumCircuit(1)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rz(self.weight_param, 0)

        # Observable (Pauli Z)
        self.observable = SparsePauliOp.from_list([("Z", 1)])

        # State‑vector estimator
        self.estimator = QiskitEstimator()
        self.qparams = nn.Parameter(torch.randn(1, device=self.device))

    def _quantum_output(self, x: torch.Tensor) -> torch.Tensor:
        """Return the expectation value of the circuit for a single input."""
        binding = {
            "x": x.item(),
            "w": self.qparams.item(),
        }
        # Run the circuit with the current parameters
        result = self.estimator.run(
            circuits=[self.circuit],
            parameter_bindings=[binding],
            observables=[self.observable],
        ).result()
        val = result.values[0]
        return torch.tensor(val, dtype=torch.float32, device=self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns the sum of the classical network output
        and the quantum expectation value for each sample.
        """
        cls_out = self.classical_net(inputs)
        q_out = torch.stack([self._quantum_output(x) for x in inputs], dim=0)
        return cls_out + q_out.unsqueeze(-1)

    def train_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        loss_fn=nn.MSELoss(),
        early_stop_patience: int = 10,
    ) -> None:
        """
        Train the hybrid model using a standard PyTorch training loop.
        """
        dataset = TensorDataset(
            torch.from_numpy(X).float().to(self.device),
            torch.from_numpy(y).float().to(self.device),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = Adam(self.parameters(), lr=lr)
        best_loss = float("inf")
        patience = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print("Early stopping triggered.")
                    break
