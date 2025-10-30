import torch
from torch import nn
import numpy as np
from quantum_fcl import HybridFullyConnected as QuantumCircuit

class HybridFullyConnected(nn.Module):
    """
    Hybrid classical‑quantum fully connected layer.
    Combines a linear transformation with a parameterised quantum circuit.
    The quantum parameters are optimised via finite‑difference gradients.
    """

    def __init__(self,
                 n_features: int = 1,
                 n_qubits: int = 1,
                 n_params: int = 1,
                 device: str = "cpu",
                 learning_rate: float = 1e-3):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(n_features, 1).to(device)
        # Instantiate the quantum circuit
        self.qc = QuantumCircuit(n_qubits, n_params)
        # Initialise quantum parameters as torch parameters
        init_params = np.random.randn(n_params)
        self.q_params = nn.Parameter(torch.tensor(init_params,
                                                  dtype=torch.float32,
                                                  device=device))
        # Optimiser for classical weights and quantum parameters
        self.optimizer = torch.optim.Adam([*self.linear.parameters(), self.q_params],
                                          lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical linear output + quantum expectation.
        """
        x = x.to(self.device)
        classical = self.linear(x)
        # Compute quantum expectation (no autograd)
        with torch.no_grad():
            q_out = self.qc.run(self.q_params.detach().cpu().numpy())
        # Broadcast quantum output to batch dimension
        q_out_torch = torch.tensor(q_out,
                                   dtype=torch.float32,
                                   device=self.device)
        return classical + q_out_torch

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        One optimisation step: back‑prop classical part, update quantum params
        with a finite‑difference gradient.
        """
        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        # Update classical weights
        self.optimizer.step()
        # Finite‑difference update for quantum parameters
        q_grad = self._finite_diff_grad(x, y, pred)
        self.q_params.data -= 0.01 * torch.tensor(q_grad,
                                                   dtype=torch.float32,
                                                   device=self.device)
        return loss

    def _finite_diff_grad(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          pred: torch.Tensor) -> np.ndarray:
        """
        Compute the gradient of the loss w.r.t quantum parameters
        using a central finite‑difference approximation.
        """
        eps = 1e-4
        grads = np.zeros_like(self.q_params.data.cpu().numpy())
        loss_fn = lambda out: nn.functional.mse_loss(out, y).item()
        # Classical output (detached)
        classical_out = self.linear(x).detach().cpu().numpy()
        for i in range(len(self.q_params)):
            perturbed = self.q_params.data.cpu().numpy().copy()
            perturbed[i] += eps
            q_out_plus = self.qc.run(perturbed)
            pred_plus = torch.tensor(classical_out + q_out_plus,
                                     dtype=torch.float32,
                                     device=self.device)
            loss_plus = loss_fn(pred_plus)
            perturbed[i] -= 2 * eps
            q_out_minus = self.qc.run(perturbed)
            pred_minus = torch.tensor(classical_out + q_out_minus,
                                      dtype=torch.float32,
                                      device=self.device)
            loss_minus = loss_fn(pred_minus)
            grads[i] = (loss_plus - loss_minus) / (2 * eps)
        return grads
