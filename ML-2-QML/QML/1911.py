import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quantum states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Labels are derived from the parameters with optional Gaussian noise.
    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the labels.
    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_wires)
        Complex amplitude vectors of the quantum states.
    labels : np.ndarray of shape (samples,)
        Target values (real numbers).
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
    if noise_std > 0.0:
        labels += noise_std * np.random.randn(samples)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Torch Dataset wrapping the synthetic quantum states.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class QModel(tq.QuantumModule):
    """
    Quantum regression model with a variational circuit and an optional quantum Fisher information estimate.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.num_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        # Encode the input state into the device
        # Using a simple Ry rotation per wire as in the original seed
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Construct a stack of variational layers
        self.layers = nn.ModuleList([self.QLayer(num_wires) for _ in range(n_layers)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a scalar regression output per sample.
        Parameters
        ----------
        state_batch : torch.Tensor of shape (batch, 2**num_wires)
            Batch of quantum states in statevector representation.
        Returns
        -------
        output : torch.Tensor of shape (batch,)
            Regression predictions.
        """
        batch_size = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=batch_size, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        features = self.measure(qdev)  # shape (batch, num_wires)
        return self.head(features).squeeze(-1)

    def qfi(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Approximate quantum Fisher information (QFI) for the variational circuit
        by computing the sum of squared gradients of the output w.r.t. all trainable parameters.
        This provides an estimate of the sensitivity of the circuit to its parameters.
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of input states.
        Returns
        -------
        qfi_est : torch.Tensor of shape (batch,)
            Estimated QFI per sample.
        """
        self.requires_grad_(True)
        output = self.forward(state_batch)
        grads = torch.autograd.grad(
            outputs=output,
            inputs=self.parameters(),
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )
        # Sum of squared gradients across all parameters
        qfi_est = sum((g ** 2).sum(dim=1) for g in grads if g is not None)
        return qfi_est

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
